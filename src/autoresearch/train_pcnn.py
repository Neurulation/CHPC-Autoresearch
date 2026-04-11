"""Training script for Predictive Coding networks (PC-FFNN and variants).

PC networks require a two-phase training loop:
  Phase 1 — Inference: minimise free energy F over {r_l} (weights fixed, T_pc steps).
             Supervised: clamp r_L = one_hot(y).
  Phase 2 — Weight update: minimise F over {W_l} (representations fixed).
             Implemented as backprop on F w.r.t. weights via Adam.

Training loss:  PC free energy F = 0.5 * Σ_l ||r_l - f(W_l @ r_{l-1})||²
Validation loss: Cross-entropy on free-inference logits (r_L; no clamping).
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import torch
import torch.nn as nn
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from autoresearch.utils import (
    create_dataloaders,
    initialize_wandb,
    set_seed,
    setup_device,
)

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers (shared with train.py)
# ---------------------------------------------------------------------------

def setup_run_directory(cfg: DictConfig) -> Path:
    run_dir = Path("outputs") / "train" / cfg.experiment_group / f"{cfg.seed}"
    (run_dir / "checkpoints").mkdir(exist_ok=True, parents=True)
    config_path = run_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    log.info("Run directory: %s", run_dir)
    return run_dir


def save_status(
    run_dir: Path,
    status: str,
    epoch: Optional[int] = None,
    best_val_loss: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    status_data = {
        "status": status,
        "last_updated": datetime.now().isoformat(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "message": message,
    }
    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)


def check_status(run_dir: Path) -> Optional[Dict[str, Any]]:
    status_file = run_dir / "status.json"
    if status_file.exists():
        with open(status_file, encoding="utf-8") as f:
            return json.load(f)
    return None


def save_checkpoint(
    checkpoint_dir: Path,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    config: DictConfig,
    wandb_run_id: Optional[str] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_metric": best_metric,
        "config": OmegaConf.to_container(config, resolve=True),
        "rng_state": {
            "torch": torch.get_rng_state(),
            "cuda": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        "wandb_run_id": wandb_run_id,
    }
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_dir / "last_checkpoint.pt")
    log.info("Saved checkpoint (epoch %d)", epoch)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    log.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if "rng_state" in checkpoint:
        rng = checkpoint["rng_state"]["torch"]
        if not isinstance(rng, torch.ByteTensor):
            rng = rng.byte()
        if rng.device.type != "cpu":
            rng = rng.cpu()
        torch.set_rng_state(rng)
        if checkpoint["rng_state"]["cuda"] is not None and torch.cuda.is_available():
            cuda_rng = checkpoint["rng_state"]["cuda"]
            if isinstance(cuda_rng, list):
                cuda_rng = [
                    s.byte().cpu() if not isinstance(s, torch.ByteTensor) or s.device.type != "cpu" else s
                    for s in cuda_rng
                ]
            torch.cuda.set_rng_state_all(cuda_rng)
    log.info("Resumed from epoch %d", checkpoint["epoch"])
    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "wandb_run_id": checkpoint.get("wandb_run_id"),
    }


def save_best_model(
    checkpoint_dir: Path, epoch: int, model: nn.Module, val_loss: float
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for old in checkpoint_dir.glob("best_model_*.pt"):
        old.unlink()
    torch.save(
        {"model_state_dict": model.state_dict(), "epoch": epoch, "val_loss": val_loss},
        checkpoint_dir / "best_model.pt",
    )
    log.info("Saved best model (epoch %d, val_loss %.4f)", epoch, val_loss)


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, metric: float) -> bool:
        score = -metric if self.mode == "min" else metric
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.should_stop


# ---------------------------------------------------------------------------
# PC-specific train / validate
# ---------------------------------------------------------------------------

def train_one_epoch_pc(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    wandb_enabled: bool = False,
    log_frequency: int = 10,
) -> Dict[str, float]:
    """One PC training epoch.

    Calls model.pc_loss(x, y) which runs supervised inference then returns
    the PC energy (training loss) and logits (for accuracy logging).
    """
    model.train()
    total_energy = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        energy, logits = model.pc_loss(x, y)
        energy.backward()
        optimizer.step()

        total_energy += energy.item()
        _, predicted = torch.max(logits, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        pbar.set_postfix({"energy": energy.item()})

        if wandb_enabled and (batch_idx + 1) % log_frequency == 0:
            import wandb
            wandb.log({"train/energy_step": energy.item(), "epoch": epoch})

    return {
        "loss": total_energy / len(train_loader),   # PC energy (training loss)
        "accuracy": 100.0 * correct / total,
    }


def validate_pc(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate using free inference.

    Reports cross-entropy on inferred logits (r_L) for apples-to-apples
    comparison with other models in the leaderboard.
    """
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)          # free inference → r_L as logits
            loss = ce_fn(logits, y)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return {
        "loss": total_loss / len(val_loader),   # CE (for comparison)
        "accuracy": 100.0 * correct / total,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    cfg: DictConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    start_epoch: int,
    best_metric: float,
    wandb_run_id: Optional[str],
    wandb_enabled: bool,
    run_dir: Path,
) -> None:
    """PC training loop."""
    checkpoint_dir = run_dir / "checkpoints"

    save_status(run_dir, "in_progress", epoch=start_epoch, best_val_loss=best_metric,
                message=f"Training started from epoch {start_epoch}")

    early_stopping = None
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode=cfg.early_stopping.mode,
        )
        log.info("Early stopping: patience=%d", cfg.early_stopping.patience)

    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_metrics = train_one_epoch_pc(
                model, train_loader, optimizer, device=next(model.parameters()).device,
                epoch=epoch, wandb_enabled=wandb_enabled,
                log_frequency=cfg.wandb.log_frequency,
            )
            val_metrics = validate_pc(
                model, val_loader, device=next(model.parameters()).device
            )

            log.info(
                "Epoch %d/%d  Train Energy: %.4f  Train Acc: %.2f%%  "
                "Val CE: %.4f  Val Acc: %.2f%%",
                epoch, cfg.epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
            )

            if wandb_enabled:
                import wandb
                wandb.log({
                    "train/energy": train_metrics["loss"],
                    "train/accuracy": train_metrics["accuracy"],
                    "val/loss": val_metrics["loss"],
                    "val/accuracy": val_metrics["accuracy"],
                    "epoch": epoch,
                })

            if cfg.checkpoint.enabled and epoch % cfg.checkpoint.save_frequency == 0:
                save_checkpoint(
                    checkpoint_dir, epoch, model, optimizer,
                    min(best_metric, val_metrics["loss"]), cfg, wandb_run_id,
                )

            if cfg.checkpoint.enabled and val_metrics["loss"] < best_metric:
                best_metric = val_metrics["loss"]
                save_best_model(checkpoint_dir, epoch, model, val_metrics["loss"])

            if early_stopping is not None and early_stopping(val_metrics["loss"]):
                log.info("Early stopping at epoch %d", epoch)
                save_status(run_dir, "completed", epoch=epoch, best_val_loss=best_metric,
                            message=f"Early stopping triggered at epoch {epoch}")
                return

        save_status(run_dir, "completed", epoch=cfg.epochs, best_val_loss=best_metric,
                    message="Training completed successfully")

    except Exception as e:
        log.error("Training failed: %s", e)
        save_status(run_dir, "failed", message=str(e))
        raise

    finally:
        if wandb_enabled:
            import wandb
            wandb.finish()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="train_pcnn")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    run_dir = setup_run_directory(cfg)

    status = check_status(run_dir)
    if status is not None:
        if status["status"] == "completed":
            log.warning("Experiment already completed at epoch %s. Delete status.json to re-run.",
                        status.get("epoch", "?"))
            return
        log.info("Resuming (%s) from epoch %s", status["status"], status.get("epoch", "?"))

    device = setup_device()

    # Data
    log.info("Loading dataset...")
    dataset = instantiate(cfg.dataset, train=True)
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Model
    model = instantiate(cfg.model)
    model = model.to(device)
    log.info("Model: %s", model)
    num_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d", num_params)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Resume from checkpoint if available
    start_epoch = 1
    best_metric = float("inf")
    wandb_run_id = None
    checkpoint_dir = run_dir / "checkpoints"
    last_ckpt = checkpoint_dir / "last_checkpoint.pt"
    if last_ckpt.exists():
        info = load_checkpoint(last_ckpt, model, optimizer, device)
        start_epoch = info["start_epoch"]
        best_metric = info["best_metric"]
        wandb_run_id = info["wandb_run_id"]
    elif cfg.checkpoint.enabled and cfg.checkpoint.save_initial_model:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(),
                    "config": OmegaConf.to_container(cfg, resolve=True)},
                   checkpoint_dir / "initial_model.pt")

    # Evaluate initial model
    if start_epoch == 1:
        log.info("Evaluating initial model (epoch 0)...")
        init_val = validate_pc(model, val_loader, device)
        log.info("Epoch 0 — Val CE: %.4f  Val Acc: %.2f%%",
                 init_val["loss"], init_val["accuracy"])

    # W&B
    wandb_enabled, wandb_run_id = initialize_wandb(
        cfg, run_dir, wandb_run_id,
        {"num_model_params": num_params, "T_pc": cfg.model.T_pc, "lr_pc": cfg.model.lr_pc},
    )

    # Train
    train(
        cfg, model, train_loader, val_loader, optimizer,
        start_epoch, best_metric, wandb_run_id, wandb_enabled, run_dir,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

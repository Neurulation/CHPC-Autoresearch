"""Training script for contrastive self-supervised learning (SimCLR-style).

Two-phase training loop:

  Phase 1 — Contrastive pre-training (unsupervised):
    - The full (unlabelled) MNIST training set is used.
    - Each image is augmented twice to produce a positive pair (x1, x2).
    - The model's encoder + projection_head are trained with NT-Xent loss.
    - Optional: PC energy regularisation for PCContrastive models.

  Phase 2 — Linear probing (supervised):
    - The encoder is *frozen*.
    - Only model.linear_probe is trained with standard CE loss.
    - Reports the primary evaluation metric: linear probe accuracy.

Both phases write to the same output directory, allowing full resume.

Entry point:
    python -m autoresearch.train_contrastive experiment=mnist_simclr_ffnn
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import torch
import torch.nn as nn
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from autoresearch.utils import (
    MNISTContrastiveAugmentation,
    initialize_wandb,
    nt_xent_loss,
    set_seed,
    setup_device,
    validate,
)

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers shared with train.py / train_pcnn.py
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
    phase: str,
    wandb_run_id: Optional[str] = None,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "phase": phase,
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
    torch.save(checkpoint, checkpoint_dir / f"checkpoint_{phase}_epoch_{epoch}.pt")
    torch.save(checkpoint, checkpoint_dir / f"last_checkpoint_{phase}.pt")
    log.info("Saved %s checkpoint (epoch %d)", phase, epoch)


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
                    s.byte().cpu()
                    if not isinstance(s, torch.ByteTensor) or s.device.type != "cpu"
                    else s
                    for s in cuda_rng
                ]
            torch.cuda.set_rng_state_all(cuda_rng)
    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "wandb_run_id": checkpoint.get("wandb_run_id"),
        "phase": checkpoint.get("phase", "pretrain"),
    }


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
# Contrastive dataset wrapper
# ---------------------------------------------------------------------------

class ContrastiveDataset(torch.utils.data.Dataset):
    """Wraps a dataset to return two augmented views of each image.

    Args:
        dataset:    Base dataset returning (image, label) tuples.
        augmentation: MNISTContrastiveAugmentation instance (or any
                      callable returning (view1, view2) from one image).
    """

    def __init__(self, dataset, augmentation):
        self.dataset = dataset
        self.augmentation = augmentation

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        x1, x2 = self.augmentation(x)
        return x1, x2, y


# ---------------------------------------------------------------------------
# Phase 1: Contrastive pre-training
# ---------------------------------------------------------------------------

def pretrain_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    temperature: float,
    wandb_enabled: bool,
    log_frequency: int,
    max_grad_norm: Optional[float],
    energy_weight: float = 0.0,
) -> Dict[str, float]:
    """One epoch of contrastive pre-training."""
    model.train()
    total_loss = 0.0
    total_energy = 0.0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Pretrain Epoch {epoch}")
    for batch_idx, (x1, x2, _) in enumerate(pbar):
        x1, x2 = x1.to(device), x2.to(device)

        optimizer.zero_grad()

        h1 = model.encode(x1)
        h2 = model.encode(x2)
        z1 = model.project(h1)
        z2 = model.project(h2)
        loss = nt_xent_loss(z1, z2, temperature=temperature)

        # Optional PC energy regularisation (PCContrastive only)
        energy_val = 0.0
        if energy_weight > 0 and hasattr(model, "pc_energy"):
            energy = model.pc_energy(x1)
            energy_val = energy.item()
            loss = loss + energy_weight * energy

        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        total_energy += energy_val
        n_batches += 1
        pbar.set_postfix({"nt_xent": f"{loss.item():.4f}"})

        if wandb_enabled and (batch_idx + 1) % log_frequency == 0:
            import wandb
            wandb.log({
                "pretrain/nt_xent_step": loss.item(),
                "pretrain/energy_step": energy_val,
                "epoch": epoch,
            })

    return {
        "nt_xent": total_loss / n_batches,
        "energy": total_energy / n_batches,
    }


# ---------------------------------------------------------------------------
# Phase 2: Linear probing
# ---------------------------------------------------------------------------

def probe_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    wandb_enabled: bool,
    log_frequency: int,
    max_grad_norm: Optional[float],
) -> Dict[str, float]:
    """One epoch of linear probe training (encoder frozen)."""
    model.eval()  # keep encoder in eval mode (BN uses running stats)
    model.linear_probe.train()

    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    pbar = tqdm(loader, desc=f"Probe Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        with torch.no_grad():
            h = model.encode(x)
        logits = model.classify(h)
        loss = ce_fn(logits, y)
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.linear_probe.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
        n_batches += 1
        pbar.set_postfix({"ce": f"{loss.item():.4f}"})

        if wandb_enabled and (batch_idx + 1) % log_frequency == 0:
            import wandb
            wandb.log({"probe/ce_step": loss.item(), "epoch": epoch})

    return {
        "loss": total_loss / n_batches,
        "accuracy": 100.0 * correct / total,
    }


def probe_validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate linear probe (encoder frozen)."""
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            h = model.encode(x)
            logits = model.classify(h)
            loss = ce_fn(logits, y)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    return {
        "loss": total_loss / len(loader),
        "accuracy": 100.0 * correct / total,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def run_pretrain(
    cfg: DictConfig,
    model: nn.Module,
    pretrain_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    start_epoch: int,
    best_metric: float,
    wandb_run_id: Optional[str],
    wandb_enabled: bool,
    run_dir: Path,
    scheduler=None,
) -> None:
    """Run contrastive pre-training phase."""
    checkpoint_dir = run_dir / "checkpoints"
    augmentation = MNISTContrastiveAugmentation(
        crop_size=cfg.augmentation.crop_size,
        padding=cfg.augmentation.padding,
        flip_p=cfg.augmentation.flip_p,
        brightness=cfg.augmentation.brightness,
        contrast=cfg.augmentation.contrast,
        noise_std=cfg.augmentation.noise_std,
        erasing_p=cfg.augmentation.erasing_p,
    )
    contrastive_loader = DataLoader(
        ContrastiveDataset(pretrain_loader.dataset, augmentation),
        batch_size=cfg.pretrain_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    early_stopping = None
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode="min",
        )

    best_nt_xent = best_metric

    try:
        for epoch in range(start_epoch, cfg.pretrain_epochs + 1):
            train_metrics = pretrain_one_epoch(
                model, contrastive_loader, optimizer,
                device=next(model.parameters()).device,
                epoch=epoch,
                temperature=cfg.temperature,
                wandb_enabled=wandb_enabled,
                log_frequency=cfg.wandb.log_frequency,
                max_grad_norm=cfg.get("max_grad_norm", None),
                energy_weight=cfg.get("energy_weight", 0.0),
            )
            if scheduler is not None:
                scheduler.step()

            log.info(
                "Pretrain Epoch %d/%d  NT-Xent: %.4f  Energy: %.4f",
                epoch, cfg.pretrain_epochs,
                train_metrics["nt_xent"], train_metrics["energy"],
            )

            if wandb_enabled:
                import wandb
                log_dict = {
                    "pretrain/nt_xent": train_metrics["nt_xent"],
                    "pretrain/energy": train_metrics["energy"],
                    "epoch": epoch,
                }
                if scheduler is not None:
                    log_dict["pretrain/lr"] = scheduler.get_last_lr()[0]
                wandb.log(log_dict)

            if cfg.checkpoint.enabled and epoch % cfg.checkpoint.save_frequency == 0:
                save_checkpoint(
                    checkpoint_dir, epoch, model, optimizer,
                    train_metrics["nt_xent"], cfg, "pretrain", wandb_run_id,
                )
                if train_metrics["nt_xent"] < best_nt_xent:
                    best_nt_xent = train_metrics["nt_xent"]

            if early_stopping is not None and early_stopping(train_metrics["nt_xent"]):
                log.info("Early stopping pretrain at epoch %d", epoch)
                break

        save_status(run_dir, "pretrain_completed", epoch=epoch,
                    best_val_loss=best_nt_xent,
                    message=f"Pre-training completed at epoch {epoch}")
        # Save final pretrain model
        torch.save(
            {"model_state_dict": model.state_dict(), "epoch": epoch},
            checkpoint_dir / "pretrain_final.pt",
        )

    except Exception as e:
        log.error("Pre-training failed: %s", e)
        save_status(run_dir, "failed", message=str(e))
        raise


def run_probe(
    cfg: DictConfig,
    model: nn.Module,
    probe_loader: DataLoader,
    val_loader: DataLoader,
    run_dir: Path,
    wandb_enabled: bool,
    wandb_run_id: Optional[str],
) -> None:
    """Run linear probe phase (encoder frozen)."""
    checkpoint_dir = run_dir / "checkpoints"
    device = next(model.parameters()).device

    # Freeze encoder — only train linear_probe
    for param in model.parameters():
        param.requires_grad_(False)
    for param in model.linear_probe.parameters():
        param.requires_grad_(True)

    probe_optimizer = torch.optim.Adam(
        model.linear_probe.parameters(), lr=cfg.probe_lr
    )

    early_stopping = None
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode="max",
        )

    best_probe_acc = 0.0

    try:
        for epoch in range(1, cfg.probe_epochs + 1):
            train_metrics = probe_one_epoch(
                model, probe_loader, probe_optimizer, device,
                epoch, wandb_enabled, cfg.wandb.log_frequency,
                cfg.get("max_grad_norm", None),
            )
            val_metrics = probe_validate(model, val_loader, device)

            log.info(
                "Probe Epoch %d/%d  Train CE: %.4f  Train Acc: %.2f%%  "
                "Val CE: %.4f  Val Acc: %.2f%%",
                epoch, cfg.probe_epochs,
                train_metrics["loss"], train_metrics["accuracy"],
                val_metrics["loss"], val_metrics["accuracy"],
            )

            if wandb_enabled:
                import wandb
                wandb.log({
                    "probe/train_loss": train_metrics["loss"],
                    "probe/train_accuracy": train_metrics["accuracy"],
                    "probe/val_loss": val_metrics["loss"],
                    "probe/val_accuracy": val_metrics["accuracy"],
                    "probe_epoch": epoch,
                })

            if cfg.checkpoint.enabled and val_metrics["accuracy"] > best_probe_acc:
                best_probe_acc = val_metrics["accuracy"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "probe_val_acc": best_probe_acc,
                    },
                    checkpoint_dir / "best_probe_model.pt",
                )
                log.info("New best probe val_acc: %.2f%% (epoch %d)",
                         best_probe_acc, epoch)

            if early_stopping is not None and early_stopping(val_metrics["accuracy"]):
                log.info("Early stopping probe at epoch %d (best: %.2f%%)",
                         epoch, best_probe_acc)
                break

        save_status(run_dir, "completed", epoch=epoch,
                    best_val_loss=val_metrics["loss"],
                    message=f"Probe completed. Best probe val_acc: {best_probe_acc:.2f}%")
        log.info("Final best probe val_acc: %.2f%%", best_probe_acc)

    except Exception as e:
        log.error("Probe phase failed: %s", e)
        save_status(run_dir, "failed", message=str(e))
        raise

    finally:
        if wandb_enabled:
            import wandb
            wandb.finish()

    # Re-enable gradients for all params (clean state)
    for param in model.parameters():
        param.requires_grad_(True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="train_contrastive")
def main(cfg: DictConfig) -> None:
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)

    run_dir = setup_run_directory(cfg)

    status = check_status(run_dir)
    if status is not None:
        if status["status"] == "completed":
            log.warning(
                "Experiment already completed. Delete status.json to re-run."
            )
            return
        log.info("Resuming (%s) from epoch %s",
                 status["status"], status.get("epoch", "?"))

    device = setup_device()

    # Data: full train set (unsupervised for pre-training)
    log.info("Loading dataset...")
    dataset = instantiate(cfg.dataset, train=True)

    # 90/10 split for probe training/validation
    n_val = int(0.1 * len(dataset))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )

    pretrain_loader = DataLoader(
        train_dataset,
        batch_size=cfg.pretrain_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    probe_loader = DataLoader(
        train_dataset,
        batch_size=cfg.probe_batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.probe_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    model = instantiate(cfg.model).to(device)
    log.info("Model: %s", model)
    num_params = sum(p.numel() for p in model.parameters())
    log.info("Parameters: %d", num_params)

    # Optimiser for pre-training (encoder + projection head)
    pretrain_optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # Resume from checkpoint if available
    start_epoch = 1
    best_metric = float("inf")
    wandb_run_id = None
    checkpoint_dir = run_dir / "checkpoints"
    last_ckpt = checkpoint_dir / "last_checkpoint_pretrain.pt"

    pretrain_done = (checkpoint_dir / "pretrain_final.pt").exists()

    if not pretrain_done and last_ckpt.exists():
        info = load_checkpoint(last_ckpt, model, pretrain_optimizer, device)
        start_epoch = info["start_epoch"]
        best_metric = info["best_metric"]
        wandb_run_id = info["wandb_run_id"]

    # W&B
    wandb_enabled, wandb_run_id = initialize_wandb(
        cfg, run_dir, wandb_run_id,
        {"num_model_params": num_params},
    )

    # LR scheduler (optional)
    scheduler = None
    if cfg.get("scheduler") is not None:
        scheduler = instantiate(cfg.scheduler, optimizer=pretrain_optimizer,
                                T_max=cfg.pretrain_epochs)

    # Phase 1 — Contrastive pre-training
    if not pretrain_done:
        save_status(run_dir, "pretraining", epoch=start_epoch,
                    message="Contrastive pre-training started")
        run_pretrain(
            cfg, model, pretrain_loader, val_loader,
            pretrain_optimizer, start_epoch, best_metric,
            wandb_run_id, wandb_enabled, run_dir, scheduler=scheduler,
        )
    else:
        log.info("Pre-training already completed; loading pretrain_final.pt")
        ckpt = torch.load(checkpoint_dir / "pretrain_final.pt", map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    # Phase 2 — Linear probe
    save_status(run_dir, "probing", message="Linear probe phase started")
    run_probe(cfg, model, probe_loader, val_loader, run_dir,
              wandb_enabled, wandb_run_id)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

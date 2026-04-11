"""Training script with Hydra, vanilla PyTorch, Wandb, and resume capability."""

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
    validate,
)

load_dotenv()

log = logging.getLogger(__name__)


def setup_run_directory(cfg: DictConfig) -> Path:
    """Setup and configure the run directory for this experiment.

    Args:
        cfg: Configuration object

    Returns:
        Path to the run directory
    """
    run_dir = Path("outputs") / "train" / cfg.experiment_group / f"{cfg.seed}"

    # Create subdirectories
    (run_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

    # Save the full resolved config
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
    """Save status file to track experiment state.

    Args:
        run_dir: Run directory
        status: Status string ('in_progress', 'completed', 'failed')
        epoch: Current epoch number
        best_val_loss: Best validation loss
        message: Additional status message
    """
    status_file = run_dir / "status.json"
    status_data = {
        "status": status,
        "last_updated": datetime.now().isoformat(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "message": message,
    }
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)


def check_status(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Check if experiment is already completed or in progress.

    Args:
        run_dir: Run directory

    Returns:
        Status dictionary if exists, None otherwise
    """
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
    """Save a checkpoint with all training state.

    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        model: Model to save
        optimizer: Optimizer to save
        best_metric: Best validation metric so far
        config: Training configuration
        wandb_run_id: Wandb run ID for resuming
    """
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

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    log.info("Saved checkpoint to %s", checkpoint_path)

    # Save as last checkpoint
    last_checkpoint_path = checkpoint_dir / "last_checkpoint.pt"
    torch.save(checkpoint, last_checkpoint_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, Any]:
    """Load checkpoint and restore training state.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into
        device: Device to load checkpoint on

    Returns:
        Dictionary with checkpoint metadata
    """
    log.info("Loading checkpoint from %s", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore RNG states
    if "rng_state" in checkpoint:
        torch_rng_state = checkpoint["rng_state"]["torch"]
        if not isinstance(torch_rng_state, torch.ByteTensor):
            torch_rng_state = torch_rng_state.byte()
        if torch_rng_state.device.type != "cpu":
            torch_rng_state = torch_rng_state.cpu()
        torch.set_rng_state(torch_rng_state)

        if checkpoint["rng_state"]["cuda"] is not None and torch.cuda.is_available():
            cuda_rng_state = checkpoint["rng_state"]["cuda"]
            if isinstance(cuda_rng_state, list):
                cuda_rng_state = [
                    state.byte().cpu()
                    if not isinstance(state, torch.ByteTensor)
                    or state.device.type != "cpu"
                    else state
                    for state in cuda_rng_state
                ]
            torch.cuda.set_rng_state_all(cuda_rng_state)

    log.info("Resumed from epoch %s", checkpoint["epoch"])

    return {
        "start_epoch": checkpoint["epoch"] + 1,
        "best_metric": checkpoint.get("best_metric", float("inf")),
        "wandb_run_id": checkpoint.get("wandb_run_id"),
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    wandb_enabled: bool = False,
    log_frequency: int = 10,
    max_grad_norm: Optional[float] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        loss_fn: Loss function
        device: Device to train on
        epoch: Current epoch number
        wandb_enabled: Whether to log to W&B
        log_frequency: How often to log to W&B

    Returns:
        Dictionary with average metrics for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Forward pass
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        # Backward pass
        loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = torch.max(y_pred.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

        # Log to W&B (step-level metrics)
        if wandb_enabled and (batch_idx + 1) % log_frequency == 0:
            import wandb

            wandb.log(
                {
                    "train/loss_step": loss.item(),
                    "epoch": epoch,
                }
            )

    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return {"loss": avg_loss, "accuracy": accuracy}


class EarlyStopping:
    """Early stopping handler."""

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


def setup_data(cfg: DictConfig) -> Tuple[DataLoader, DataLoader]:
    """Setup dataset and dataloaders.

    Args:
        cfg: Configuration object

    Returns:
        Tuple of (train_loader, val_loader)
    """
    log.info("Loading dataset...")
    dataset = instantiate(cfg.dataset, train=True)
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    log.info("Training samples: %s", len(train_loader.dataset))
    log.info("Validation samples: %s", len(val_loader.dataset))
    return train_loader, val_loader


def setup_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    """Create and setup model.

    Args:
        cfg: Configuration object
        device: Device to move model to

    Returns:
        Initialized model
    """
    log.info("Creating model...")
    model = instantiate(cfg.model)
    model = model.to(device)
    log.info("Model: %s", model)
    return model


def get_resume_path(run_dir: Path) -> Optional[Path]:
    """Determine the checkpoint path to resume from.

    Args:
        run_dir: Run directory

    Returns:
        Path to checkpoint or None
    """
    checkpoint_dir = run_dir / "checkpoints"
    resume_path = None

    if checkpoint_dir.exists() and (checkpoint_dir / "last_checkpoint.pt").exists():
        log.info("Found existing checkpoint, resuming...")
        resume_path = checkpoint_dir / "last_checkpoint.pt"

    return resume_path


def save_best_model(
    checkpoint_dir: Path, epoch: int, model: nn.Module, val_loss: float
) -> None:
    """Save the best model checkpoint.

    Args:
        checkpoint_dir: Directory to save checkpoint
        epoch: Current epoch number
        model: Model to save
        val_loss: Validation loss value
    """
    best_model_path = checkpoint_dir / "best_model.pt"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Remove old best models with metric in name
    for old_best in checkpoint_dir.glob("best_model_*.pt"):
        old_best.unlink()

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        best_model_path,
    )
    log.info(
        "Saved best model to %s (epoch %d, val_loss %.4f)",
        best_model_path,
        epoch,
        val_loss,
    )


def save_initial_model(
    checkpoint_dir: Path, model: nn.Module, config: DictConfig
) -> None:
    """Save the initial model state before training.

    Args:
        checkpoint_dir: Directory to save checkpoint
        model: Model with random initialization
        config: Training configuration
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_model_path = checkpoint_dir / "initial_model.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": OmegaConf.to_container(config, resolve=True),
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
        },
        initial_model_path,
    )
    log.info("Saved initial model state to %s", initial_model_path)


def log_epoch_metrics(
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    wandb_enabled: bool,
) -> None:
    """Log metrics for the current epoch."""
    log.info(
        "Epoch %s/%s - Train Loss: %.4f, Train Acc: %.2f%%, Val Loss: %.4f, Val Acc: %.2f%%",
        epoch,
        total_epochs,
        train_metrics["loss"],
        train_metrics["accuracy"],
        val_metrics["loss"],
        val_metrics["accuracy"],
    )

    if wandb_enabled:
        import wandb

        wandb.log(
            {
                "train/loss": train_metrics["loss"],
                "train/accuracy": train_metrics["accuracy"],
                "val/loss": val_metrics["loss"],
                "val/accuracy": val_metrics["accuracy"],
                "epoch": epoch,
            }
        )


def train(
    cfg: DictConfig,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    start_epoch: int,
    best_metric: float,
    wandb_run_id: Optional[str],
    wandb_enabled: bool,
    run_dir: Path,
    scheduler: Optional[Any] = None,
) -> None:
    """Run the main training loop."""
    checkpoint_dir = run_dir / "checkpoints"

    save_status(
        run_dir,
        "in_progress",
        epoch=start_epoch,
        best_val_loss=best_metric,
        message=f"Training started from epoch {start_epoch}",
    )

    # Setup early stopping
    early_stopping = None
    if cfg.early_stopping.enabled:
        early_stopping = EarlyStopping(
            patience=cfg.early_stopping.patience,
            min_delta=cfg.early_stopping.min_delta,
            mode=cfg.early_stopping.mode,
        )
        log.info("Early stopping enabled with patience %s", cfg.early_stopping.patience)

    # Training loop
    log.info("Starting training...")
    try:
        for epoch in range(start_epoch, cfg.epochs + 1):
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                loss_fn,
                device,
                epoch,
                wandb_enabled,
                cfg.wandb.log_frequency,
                cfg.get("max_grad_norm"),
            )

            val_metrics = validate(model, val_loader, loss_fn, device)

            log_epoch_metrics(
                epoch, cfg.epochs, train_metrics, val_metrics, wandb_enabled
            )

            # Step LR scheduler
            if scheduler is not None:
                scheduler.step()
                if wandb_enabled:
                    import wandb

                    wandb.log(
                        {
                            "train/lr": scheduler.get_last_lr()[0],
                            "epoch": epoch,
                        }
                    )

            # Save checkpoint
            if cfg.checkpoint.enabled and epoch % cfg.checkpoint.save_frequency == 0:
                save_checkpoint(
                    checkpoint_dir,
                    epoch,
                    model,
                    optimizer,
                    min(best_metric, val_metrics["loss"]),
                    cfg,
                    wandb_run_id,
                )

            # Save best model
            if cfg.checkpoint.enabled and val_metrics["loss"] < best_metric:
                best_metric = val_metrics["loss"]
                save_best_model(checkpoint_dir, epoch, model, val_metrics["loss"])

            # Early stopping check
            if early_stopping is not None:
                if early_stopping(val_metrics["loss"]):
                    log.info("Early stopping triggered at epoch %s", epoch)
                    save_status(
                        run_dir,
                        "completed",
                        epoch=epoch,
                        best_val_loss=best_metric,
                        message=f"Early stopping triggered at epoch {epoch}",
                    )
                    break

        # Training completion
        log.info("Training completed!")
        save_status(
            run_dir,
            "completed",
            epoch=cfg.epochs,
            best_val_loss=best_metric,
            message="Training completed successfully",
        )

    except Exception as e:
        log.error("Training failed: %s", e)
        save_status(
            run_dir,
            "failed",
            message=str(e),
        )
        raise

    finally:
        if wandb_enabled:
            import wandb

            wandb.finish()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main training function."""
    log.info("Configuration:")
    log.info(OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    run_dir = setup_run_directory(cfg)

    # Check if experiment is already completed
    status = check_status(run_dir)
    if status is not None:
        if status["status"] == "completed":
            log.warning(
                "Experiment already completed at epoch %s. "
                "To re-run, delete the status.json file or the entire run directory.",
                status.get("epoch", "unknown"),
            )
            return
        elif status["status"] == "in_progress":
            log.info(
                "Resuming in-progress experiment from epoch %s",
                status.get("epoch", "unknown"),
            )
        elif status["status"] == "failed":
            log.info(
                "Resuming failed experiment. Previous error: %s",
                status.get("message", "unknown"),
            )

    # Setup components
    device = setup_device()
    train_loader, val_loader = setup_data(cfg)
    model = setup_model(cfg, device)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    loss_fn = instantiate(cfg.loss_fn)

    # Setup LR scheduler (optional)
    scheduler = None
    if cfg.get("scheduler") is not None:
        scheduler = instantiate(cfg.scheduler, optimizer=optimizer, T_max=cfg.epochs)

    # Handle checkpoint resuming
    start_epoch = 1
    best_metric = float("inf")
    wandb_run_id = None

    resume_path = get_resume_path(run_dir)
    if resume_path is not None and resume_path.exists():
        checkpoint_info = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = checkpoint_info["start_epoch"]
        best_metric = checkpoint_info["best_metric"]
        wandb_run_id = checkpoint_info["wandb_run_id"]
    else:
        if cfg.checkpoint.enabled and cfg.checkpoint.save_initial_model:
            checkpoint_dir = run_dir / "checkpoints"
            save_initial_model(checkpoint_dir, model, cfg)

    # Initialize W&B
    num_model_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    additional_config = {
        "num_model_params": num_model_params,
        "num_trainable_params": num_trainable_params,
    }
    wandb_enabled, wandb_run_id = initialize_wandb(
        cfg, run_dir, wandb_run_id, additional_config
    )

    # Evaluate initial model (epoch 0) if starting from scratch
    if start_epoch == 1:
        log.info("Evaluating initial model (epoch 0)...")
        initial_train_metrics = validate(model, train_loader, loss_fn, device)
        initial_val_metrics = validate(model, val_loader, loss_fn, device)

        log.info(
            "Epoch 0 (Initial) - Train Loss: %.4f, Train Acc: %.2f%%, Val Loss: %.4f, Val Acc: %.2f%%",
            initial_train_metrics["loss"],
            initial_train_metrics["accuracy"],
            initial_val_metrics["loss"],
            initial_val_metrics["accuracy"],
        )

        if wandb_enabled:
            import wandb

            wandb.log(
                {
                    "train/loss": initial_train_metrics["loss"],
                    "train/accuracy": initial_train_metrics["accuracy"],
                    "val/loss": initial_val_metrics["loss"],
                    "val/accuracy": initial_val_metrics["accuracy"],
                    "epoch": 0,
                }
            )

    # Run training
    train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        device,
        start_epoch,
        best_metric,
        wandb_run_id,
        wandb_enabled,
        run_dir,
        scheduler,
    )


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

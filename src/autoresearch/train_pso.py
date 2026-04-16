"""PSO training / search entry point.

Analogous to ``train.py`` but gradient-free.  Three operating modes are
supported via ``cfg.pso.mode``:

* ``benchmark``  — optimise a classic mathematical test function (Sphere,
  Rastrigin, Rosenbrock, Ackley).  No dataset or model required.
* ``hyperparams`` — use PSO to search learning-rate / architecture hyper-
  parameters for a small FFNN trained on MNIST.
* ``weights``    — use PSO to directly optimise the weight vector of a small
  FFNN on MNIST (neuroevolution-style).

Usage (local smoke test)::

    python -m autoresearch.train_pso experiment=pso_benchmark_sphere \\
        seed=0 pso.max_iter=5 wandb.enabled=false

"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from autoresearch.benchmarks import get_benchmark
from autoresearch.optimizers.pso import PSO
from autoresearch.utils import (
    create_dataloaders,
    initialize_wandb,
    set_seed,
    setup_device,
)

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Run-directory / status helpers (mirrors train.py)
# ---------------------------------------------------------------------------


def setup_run_directory(cfg: DictConfig) -> Path:
    """Create output directory and save resolved config."""
    run_dir = Path("outputs") / "pso" / cfg.experiment_group / f"{cfg.seed}"
    (run_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

    config_path = run_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))

    log.info("Run directory: %s", run_dir)
    return run_dir


def save_status(
    run_dir: Path,
    status: str,
    iteration: Optional[int] = None,
    best_fitness: Optional[float] = None,
    message: Optional[str] = None,
) -> None:
    """Persist experiment status to ``status.json``."""
    status_data = {
        "status": status,
        "last_updated": datetime.now().isoformat(),
        "iteration": iteration,
        "best_fitness": best_fitness,
        "message": message,
    }
    with open(run_dir / "status.json", "w", encoding="utf-8") as f:
        json.dump(status_data, f, indent=2)


def check_status(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Return status dict if ``status.json`` exists, else ``None``."""
    status_file = run_dir / "status.json"
    if status_file.exists():
        with open(status_file, encoding="utf-8") as f:
            return json.load(f)
    return None


# ---------------------------------------------------------------------------
# Fitness functions
# ---------------------------------------------------------------------------


def make_benchmark_fitness(cfg: DictConfig):
    """Return a fitness function for a mathematical benchmark."""
    fn, default_bounds, optimum = get_benchmark(cfg.benchmark.fn)
    log.info(
        "Benchmark: %s | n_dims=%d | known_optimum=%.4f",
        cfg.benchmark.fn,
        cfg.benchmark.n_dims,
        optimum,
    )
    return fn, default_bounds


def make_weights_fitness(cfg: DictConfig, device: torch.device):
    """Return a fitness function that sets FFNN weights from a flat vector.

    The fitness is the validation loss (lower = better).
    """
    log.info("Loading dataset for weight-PSO fitness...")
    dataset = instantiate(cfg.dataset, train=True)
    _, val_loader = create_dataloaders(
        dataset,
        batch_size=cfg.batch_size,
        validation_split=cfg.validation_split,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = instantiate(cfg.model).to(device)
    loss_fn = instantiate(cfg.loss_fn)

    # Determine total parameter count
    param_shapes = [(p.shape, p.numel()) for p in model.parameters()]
    total_params = sum(n for _, n in param_shapes)
    log.info("Weight-PSO dimensionality: %d", total_params)

    def fitness(x: np.ndarray) -> float:
        # Load flat vector into model parameters
        offset = 0
        with torch.no_grad():
            for param, (shape, numel) in zip(model.parameters(), param_shapes):
                chunk = torch.from_numpy(x[offset : offset + numel].reshape(shape)).float()
                param.copy_(chunk.to(device))
                offset += numel

        # Evaluate
        model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                total_loss += loss_fn(pred, yb).item()
        return total_loss / len(val_loader)

    return fitness, total_params


def make_hyperparams_fitness(cfg: DictConfig, device: torch.device):
    """Return a fitness function over (log10_lr, hidden_size, dropout).

    Trains a fresh FFNN for ``cfg.pso.hyperparams.epochs`` epochs and returns
    the final validation loss.
    """
    log.info("Setting up hyperparameter-PSO fitness...")
    hp_cfg = cfg.pso.hyperparams

    def fitness(x: np.ndarray) -> float:
        log10_lr = float(x[0])
        hidden_size = max(8, int(round(float(x[1]))))
        dropout = float(np.clip(x[2], 0.0, 0.9))
        lr = 10.0**log10_lr

        log.debug(
            "  Particle: lr=%.2e  hidden=%d  dropout=%.3f",
            lr,
            hidden_size,
            dropout,
        )

        # Build model
        local_model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 10),
        ).to(device)

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        loss_fn_local = nn.CrossEntropyLoss()

        # Fresh data split per evaluation
        from torchvision import datasets, transforms

        ds = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transforms.ToTensor(),
        )
        train_loader, val_loader = create_dataloaders(
            ds,
            batch_size=cfg.batch_size,
            validation_split=cfg.validation_split,
            num_workers=0,
            seed=cfg.seed,
        )

        local_model.train()
        for _ in range(hp_cfg.epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss_fn_local(local_model(xb), yb).backward()
                optimizer.step()

        local_model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                total_loss += loss_fn_local(local_model(xb), yb).item()

        return total_loss / len(val_loader)

    return fitness


# ---------------------------------------------------------------------------
# Main PSO loop
# ---------------------------------------------------------------------------


def run_pso(
    cfg: DictConfig,
    fitness_fn,
    n_dims: int,
    bounds: np.ndarray,
    run_dir: Path,
    wandb_enabled: bool,
) -> None:
    """Execute the PSO search loop and persist results."""
    pso = PSO(
        n_particles=cfg.pso.n_particles,
        n_dims=n_dims,
        bounds=bounds,
        w=cfg.pso.w,
        c1=cfg.pso.c1,
        c2=cfg.pso.c2,
        w_min=cfg.pso.get("w_min"),
        v_max_ratio=cfg.pso.get("v_max_ratio", 0.2),
        seed=cfg.seed,
    )

    save_status(run_dir, "in_progress", message="PSO started")

    try:
        best_pos, best_fit, history = pso.optimize(
            fitness_fn=fitness_fn,
            max_iter=cfg.pso.max_iter,
            convergence_tol=cfg.pso.convergence_tol,
            verbose=True,
            log_frequency=cfg.pso.get("log_frequency", 10),
        )

        log.info("PSO finished. Best fitness: %.6e", best_fit)

        # Log per-iteration history to WandB
        if wandb_enabled:
            import wandb

            for i, (bf, mf, ss) in enumerate(
                zip(history.best_fitness, history.mean_fitness, history.swarm_std)
            ):
                wandb.log(
                    {
                        "pso/best_fitness": bf,
                        "pso/mean_fitness": mf,
                        "pso/swarm_std": ss,
                        "iteration": i + 1,
                    }
                )
            wandb.log({"pso/final_best_fitness": best_fit})

        # Save results
        results = {
            "best_fitness": best_fit,
            "best_position": best_pos.tolist(),
            "n_iterations": len(history.best_fitness),
            "history": {
                "best_fitness": history.best_fitness,
                "mean_fitness": history.mean_fitness,
                "swarm_std": history.swarm_std,
            },
        }
        results_path = run_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        log.info("Results saved to %s", results_path)

        save_status(
            run_dir,
            "completed",
            iteration=len(history.best_fitness),
            best_fitness=best_fit,
            message="PSO completed successfully",
        )

    except Exception as e:
        log.error("PSO failed: %s", e)
        save_status(run_dir, "failed", message=str(e))
        raise

    finally:
        if wandb_enabled:
            import wandb

            wandb.finish()


# ---------------------------------------------------------------------------
# Hydra entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="pso")
def main(cfg: DictConfig) -> None:
    """Main PSO entry point."""
    log.info("PSO Configuration:\n%s", OmegaConf.to_yaml(cfg))

    set_seed(cfg.seed)

    run_dir = setup_run_directory(cfg)

    # Skip if already completed
    status = check_status(run_dir)
    if status and status["status"] == "completed":
        log.warning(
            "Experiment already completed (best_fitness=%.6e). "
            "Delete status.json to re-run.",
            status.get("best_fitness", float("nan")),
        )
        return

    device = setup_device()

    mode = cfg.pso.mode

    # -------------------------------------------------------------------
    # Build fitness function and search-space bounds
    # -------------------------------------------------------------------
    if mode == "benchmark":
        fitness_fn, default_bounds = make_benchmark_fitness(cfg)
        n_dims = cfg.benchmark.n_dims
        # Allow per-experiment override of bounds
        bounds_cfg = cfg.benchmark.get("bounds", default_bounds)
        bounds = np.asarray(bounds_cfg, dtype=float)
        if bounds.ndim == 1:
            bounds = np.broadcast_to(bounds, (n_dims, 2)).copy()

    elif mode == "weights":
        fitness_fn, n_dims = make_weights_fitness(cfg, device)
        # Initialise weights in a reasonable range for Xavier-style init
        weight_bound = cfg.pso.get("weight_bound", 1.0)
        bounds = np.array([-weight_bound, weight_bound], dtype=float)

    elif mode == "hyperparams":
        fitness_fn = make_hyperparams_fitness(cfg, device)
        # Search space: [log10_lr ∈ (-5,-1), hidden_size ∈ (16,512), dropout ∈ (0,0.9)]
        bounds = np.array(
            [
                [-5.0, -1.0],   # log10 learning rate
                [16.0, 512.0],  # hidden layer size
                [0.0, 0.9],     # dropout
            ],
            dtype=float,
        )
        n_dims = 3

    else:
        raise ValueError(f"Unknown PSO mode: '{mode}'. Choose benchmark | weights | hyperparams")

    # -------------------------------------------------------------------
    # Initialise WandB
    # -------------------------------------------------------------------
    wandb_enabled, _ = initialize_wandb(cfg, run_dir)

    # -------------------------------------------------------------------
    # Run PSO
    # -------------------------------------------------------------------
    run_pso(cfg, fitness_fn, n_dims, bounds, run_dir, wandb_enabled)


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

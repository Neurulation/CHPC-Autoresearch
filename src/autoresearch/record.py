"""Recording script — save per-sample layer activations from a trained SNN.

Loads the best checkpoint produced by ``train.py``, runs the model on the
full MNIST **test set** (no train/val split), and saves per-sample spike trains
and membrane potentials for every layer to disk.

Output directory
----------------
``outputs/record/{experiment_group}/{seed}/recordings/``

Files saved (all ``.pt`` tensors)
----------------------------------
spk1.pt   (N, T, 512)  hidden-layer-1 spike trains
mem1.pt   (N, T, 512)  hidden-layer-1 membrane potentials
spk2.pt   (N, T, 256)  hidden-layer-2 spike trains
mem2.pt   (N, T, 256)  hidden-layer-2 membrane potentials
spk3.pt   (N, T, 10)   output-layer   spike trains
mem3.pt   (N, T, 10)   output-layer   membrane potentials
labels.pt (N,)         ground-truth digit labels

where N = 10 000 (full MNIST test set).

These recordings are the direct input for Phase 2 (Artificial EEG dataset).

Usage
-----
    # Record from the Phase 1 SNN baseline (seed 0):
    python -m autoresearch.record experiment=mnist_snn_baseline_adam seed=0

    # Custom experiment group / seed:
    python -m autoresearch.record experiment_group=my_exp seed=2

Note: The model must have a ``forward_with_recordings`` method (see
``SNNBaseline``). A ``TypeError`` is raised if the loaded model does not
support recording.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import torch
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

import hydra

from autoresearch.utils import setup_device

load_dotenv()

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_checkpoint(experiment_group: str, seed: int) -> Path:
    """Return the path to the best model checkpoint for the given run.

    Looks for ``outputs/train/{experiment_group}/{seed}/checkpoints/best_model.pt``.

    Args:
        experiment_group: Experiment group name.
        seed:             Random seed of the run.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
    """
    ckpt = Path("outputs") / "train" / experiment_group / str(seed) / "checkpoints" / "best_model.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run `python -m autoresearch.train experiment=<name> seed=<seed>` first."
        )
    return ckpt


def _load_test_loader(cfg: DictConfig) -> DataLoader:
    """Create a DataLoader over the MNIST test set.

    Always uses ``train=False`` regardless of the dataset config, so that
    recordings are produced for the standard 10 000-sample test set.

    Args:
        cfg: Root Hydra configuration.

    Returns:
        DataLoader for the test set.
    """
    transform = transforms.ToTensor()
    test_dataset = datasets.MNIST(
        root=cfg.dataset.get("root", "./data"),
        train=False,
        download=True,
        transform=transform,
    )
    return DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _record_activations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run the model on every batch and collect per-sample recordings.

    Args:
        model:  A model exposing ``forward_with_recordings(x)``.
        loader: DataLoader over the test set.
        device: Computation device.

    Returns:
        Dict with keys ``spk1``, ``mem1``, ``spk2``, ``mem2``, ``spk3``,
        ``mem3`` (each ``(N, T, layer_size)``) and ``labels`` (``(N,)``).

    Raises:
        TypeError: If ``model`` does not have ``forward_with_recordings``.
    """
    if not hasattr(model, "forward_with_recordings"):
        raise TypeError(
            f"{type(model).__name__} does not implement forward_with_recordings(). "
            "Only models that support recording (e.g. SNNBaseline) can be used "
            "with record.py."
        )

    model.eval()

    buckets: Dict[str, list] = {k: [] for k in ("spk1", "mem1", "spk2", "mem2", "spk3", "mem3", "labels")}

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Recording"):
            imgs = imgs.to(device)
            spk_out, _, rec = model.forward_with_recordings(imgs)

            # Permute (T, N, D) → (N, T, D) and move to CPU immediately
            for key in ("spk1", "mem1", "spk2", "mem2", "spk3", "mem3"):
                buckets[key].append(rec[key].permute(1, 0, 2).cpu())
            buckets["labels"].append(labels.cpu())

    return {k: torch.cat(v, dim=0) for k, v in buckets.items()}


def _save_recordings(recordings: Dict[str, torch.Tensor], out_dir: Path) -> None:
    """Save all recording tensors to disk.

    Args:
        recordings: Dict of tensors to save.
        out_dir:    Directory to write ``.pt`` files into.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in recordings.items():
        path = out_dir / f"{name}.pt"
        torch.save(tensor, path)
        log.info("Saved %-8s %s → %s", name, tuple(tensor.shape), path)


def _log_accuracy(recordings: Dict[str, torch.Tensor]) -> None:
    """Log a quick spike-count decode accuracy as a sanity check."""
    spike_counts = recordings["spk3"].sum(dim=1)  # (N, num_classes)
    preds = spike_counts.argmax(dim=1)
    acc = (preds == recordings["labels"]).float().mean().item()
    log.info("Spike-count decode accuracy on test set: %.4f", acc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@hydra.main(version_base=None, config_path="configs", config_name="record")
def main(cfg: DictConfig) -> None:
    """Main recording function."""
    log.info("Record configuration:\n%s", OmegaConf.to_yaml(cfg))

    device = setup_device()

    # Locate and load checkpoint
    ckpt_path = _find_checkpoint(cfg.experiment_group, cfg.seed)
    log.info("Loading checkpoint: %s", ckpt_path)
    ckpt: Dict[str, Any] = torch.load(ckpt_path, map_location=device)

    # Instantiate model from config and load weights
    model = instantiate(cfg.model)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    log.info(
        "Loaded model from epoch %d (val_loss=%.4f)",
        ckpt.get("epoch", -1),
        ckpt.get("val_loss", float("nan")),
    )

    # Test data
    test_loader = _load_test_loader(cfg)
    log.info("Test set size: %d samples", len(test_loader.dataset))

    # Record
    recordings = _record_activations(model, test_loader, device)

    # Save
    run_dir = Path(".")  # Hydra sets cwd to outputs/record/{group}/{seed}
    recordings_dir = run_dir / "recordings"
    _save_recordings(recordings, recordings_dir)

    # Sanity check
    _log_accuracy(recordings)

    log.info("All recordings saved to: %s", recordings_dir.resolve())


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter

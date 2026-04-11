"""
SNN Baseline — recording script.

Loads the best trained checkpoint, runs the model on the **full MNIST test set**,
and saves per-sample spike trains and membrane potentials for every layer to
    experiments/snn_baseline/recordings/

Output files (all .pt tensors, shape documented inline)
--------------------------------------------------------
spk1.pt   (N, T, 512)   hidden-layer-1 spike trains
mem1.pt   (N, T, 512)   hidden-layer-1 membrane potentials
spk2.pt   (N, T, 256)   hidden-layer-2 spike trains
mem2.pt   (N, T, 256)   hidden-layer-2 membrane potentials
spk3.pt   (N, T, 10)    output-layer   spike trains
mem3.pt   (N, T, 10)    output-layer   membrane potentials
labels.pt (N,)          ground-truth digit labels

where N = 10 000 (full MNIST test set).

Usage
-----
    python experiments/snn_baseline/record.py [--timesteps T] [--batch-size B]
                                              [--device cpu|cuda]
"""

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.snn_baseline.model import SNNBaseline  # noqa: E402


# ---------------------------------------------------------------------------
# Rate encoding (same as train.py — keep in sync)
# ---------------------------------------------------------------------------

def rate_encode(imgs: torch.Tensor, timesteps: int) -> torch.Tensor:
    imgs_flat = imgs.view(imgs.size(0), -1)
    return torch.bernoulli(imgs_flat.unsqueeze(0).expand(timesteps, -1, -1))


# ---------------------------------------------------------------------------
# Recording loop
# ---------------------------------------------------------------------------

def record(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "snn_baseline_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            "Run train.py first."
        )

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device)
    saved_args = ckpt.get("args", {})
    model = SNNBaseline().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} "
          f"(val_acc={ckpt.get('val_acc', '?'):.4f})")

    # Prefer timesteps from saved args unless overridden
    timesteps = args.timesteps if args.timesteps is not None else saved_args.get("timesteps", 25)
    print(f"Recording with T={timesteps} timesteps")

    # Data — test set only, deterministic order
    transform = transforms.Compose([transforms.ToTensor()])
    test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)

    all_spk1, all_mem1 = [], []
    all_spk2, all_mem2 = [], []
    all_spk3, all_mem3 = [], []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Recording"):
            imgs = imgs.to(device)
            spikes = rate_encode(imgs, timesteps).to(device)

            _, _, rec = model(spikes)

            # Move to CPU immediately to avoid GPU OOM on large datasets
            all_spk1.append(rec["spk1"].permute(1, 0, 2).cpu())  # (batch, T, 512)
            all_mem1.append(rec["mem1"].permute(1, 0, 2).cpu())
            all_spk2.append(rec["spk2"].permute(1, 0, 2).cpu())  # (batch, T, 256)
            all_mem2.append(rec["mem2"].permute(1, 0, 2).cpu())
            all_spk3.append(rec["spk3"].permute(1, 0, 2).cpu())  # (batch, T, 10)
            all_mem3.append(rec["mem3"].permute(1, 0, 2).cpu())
            all_labels.append(labels.cpu())

    # Concatenate across batches → (N, T, D)
    recordings = {
        "spk1":   torch.cat(all_spk1,  dim=0),
        "mem1":   torch.cat(all_mem1,  dim=0),
        "spk2":   torch.cat(all_spk2,  dim=0),
        "mem2":   torch.cat(all_mem2,  dim=0),
        "spk3":   torch.cat(all_spk3,  dim=0),
        "mem3":   torch.cat(all_mem3,  dim=0),
        "labels": torch.cat(all_labels, dim=0),
    }

    out_dir = os.path.join(os.path.dirname(__file__), "recordings")
    os.makedirs(out_dir, exist_ok=True)

    for name, tensor in recordings.items():
        path = os.path.join(out_dir, f"{name}.pt")
        torch.save(tensor, path)
        print(f"  Saved {name:8s} {tuple(tensor.shape)} → {path}")

    # Quick sanity check: spike-count accuracy
    spike_counts = recordings["spk3"].sum(dim=1)  # (N, 10)
    preds = spike_counts.argmax(dim=1)
    acc = (preds == recordings["labels"]).float().mean().item()
    print(f"\nRecording accuracy (spike-count decode): {acc:.4f}")
    print(f"All recordings saved to: {out_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Record SNN Baseline activations on MNIST test set")
    parser.add_argument("--timesteps",  type=int,   default=None,  help="Timesteps T (default: use value from checkpoint)")
    parser.add_argument("--batch-size", type=int,   default=256,   help="Mini-batch size (default: 256)")
    parser.add_argument("--device",     type=str,   default="cuda", help="Device: cuda or cpu (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    record(parse_args())

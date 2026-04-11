"""
SNN Baseline — training script.

Trains the SNNBaseline model on MNIST using BPTT with snntorch surrogate
gradients, then saves the best checkpoint to
    experiments/snn_baseline/checkpoints/snn_baseline_best.pth

Usage
-----
    python experiments/snn_baseline/train.py [--epochs N] [--timesteps T]
                                             [--batch-size B] [--lr LR]
                                             [--device cpu|cuda]

Defaults are intentionally conservative so the script runs on a laptop CPU
within minutes. For a full run use --epochs 20.
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# Allow running as a standalone script from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from experiments.snn_baseline.model import SNNBaseline  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rate_encode(imgs: torch.Tensor, timesteps: int) -> torch.Tensor:
    """Convert a batch of normalised images to Poisson rate-coded spike trains.

    Parameters
    ----------
    imgs      : (batch, 1, 28, 28) float tensor in [0, 1]
    timesteps : number of time steps T

    Returns
    -------
    spikes : (T, batch, 784) binary tensor
    """
    imgs_flat = imgs.view(imgs.size(0), -1)  # (batch, 784)
    # Broadcast and sample Bernoulli with probability proportional to pixel value
    spikes = torch.bernoulli(imgs_flat.unsqueeze(0).expand(timesteps, -1, -1))
    return spikes  # (T, batch, 784)


def accuracy(spk_out: torch.Tensor, targets: torch.Tensor) -> float:
    """Decode class from spike count over all timesteps."""
    # spk_out : (T, batch, num_classes)
    spike_counts = spk_out.sum(dim=0)  # (batch, num_classes)
    preds = spike_counts.argmax(dim=1)
    return (preds == targets).float().mean().item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"Using device: {device}")

    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_set = datasets.MNIST(root="data", train=True, download=True, transform=transform)
    test_set  = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,  num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    # Model
    model = SNNBaseline().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    os.makedirs(os.path.join(os.path.dirname(__file__), "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpoints", "snn_baseline_best.pth")

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_acc  = 0.0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [train]", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            spikes = rate_encode(imgs, args.timesteps).to(device)  # (T, batch, 784)

            optimizer.zero_grad()
            spk_out, mem_out, _ = model(spikes)

            # Loss on summed membrane potential at final timestep (standard approach)
            loss = loss_fn(mem_out[-1], labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_acc  += accuracy(spk_out, labels)

        train_loss = total_loss / len(train_loader)
        train_acc  = total_acc  / len(train_loader)

        # Validation
        model.eval()
        val_acc = 0.0
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc=f"Epoch {epoch}/{args.epochs} [val]  ", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                spikes = rate_encode(imgs, args.timesteps).to(device)
                spk_out, _, _ = model(spikes)
                val_acc += accuracy(spk_out, labels)

        val_acc /= len(test_loader)
        print(f"Epoch {epoch:3d} | train loss {train_loss:.4f} | train acc {train_acc:.4f} | val acc {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, checkpoint_path)
            print(f"  → New best checkpoint saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")
    print(f"Checkpoint: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train SNN Baseline on MNIST")
    parser.add_argument("--epochs",     type=int,   default=10,    help="Number of training epochs (default: 10)")
    parser.add_argument("--timesteps",  type=int,   default=25,    help="Number of simulation timesteps T (default: 25)")
    parser.add_argument("--batch-size", type=int,   default=128,   help="Mini-batch size (default: 128)")
    parser.add_argument("--lr",         type=float, default=1e-3,  help="Adam learning rate (default: 1e-3)")
    parser.add_argument("--device",     type=str,   default="cuda", help="Device: cuda or cpu (default: cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

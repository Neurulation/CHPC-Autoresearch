"""Diagnostic: test 3 fixes for PC-FFNN energy explosion.

Based on the root cause analysis (Adam's stale v_t denominator in a
vanishing-gradient basin), we test:
  D) AdamW (weight_decay=0.01)  — keeps v_t alive via L2 gradient floor
  E) Adam with eps=0.01         — caps maximum effective step size
  F) Adam + CosineAnnealingLR   — reduces lr as energy basin narrows

Baseline (for comparison): Variant A from the initial diagnostic.

Usage:
  python scripts/diagnose_pc_fixes.py
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from autoresearch.models.pc_ffnn import PCFFNN
from autoresearch.utils.reproducibility import set_seed


def get_data(batch_size=128, val_split=0.1):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    full_train = datasets.MNIST("data", train=True, download=True, transform=transform)
    n_val = int(len(full_train) * val_split)
    n_train = len(full_train) - n_val
    train_ds, val_ds = random_split(
        full_train, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2),
    )


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = total_energy = 0.0
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        combined_loss, energy, logits = model.pc_loss(x, y)
        combined_loss.backward()
        optimizer.step()
        total_loss += combined_loss.item()
        total_energy += energy.item()
        _, pred = logits.max(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
    n = len(loader)
    return {"loss": total_loss / n, "energy": total_energy / n, "accuracy": 100.0 * correct / total}


def validate(model, loader, device):
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += ce_fn(logits, y).item()
            _, pred = logits.max(1)
            total += y.size(0)
            correct += (pred == y).sum().item()
    n = len(loader)
    return {"loss": total_loss / n, "accuracy": 100.0 * correct / total}


def run_variant(name, model, optimizer, train_loader, val_loader, device,
                epochs=20, scheduler=None):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    logs = []
    for epoch in range(1, epochs + 1):
        tm = train_one_epoch(model, train_loader, optimizer, device)
        vm = validate(model, val_loader, device)
        if scheduler is not None:
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        logs.append({
            "epoch": epoch,
            "train_energy": round(tm["energy"], 4),
            "train_loss": round(tm["loss"], 4),
            "train_acc": round(tm["accuracy"], 2),
            "val_loss": round(vm["loss"], 4),
            "val_acc": round(vm["accuracy"], 2),
            "lr": round(lr, 6),
        })
        print(f"  Epoch {epoch:2d} | E={tm['energy']:.4f} TrainAcc={tm['accuracy']:.1f}% "
              f"| ValAcc={vm['accuracy']:.2f}% ValCE={vm['loss']:.4f} lr={lr:.6f}")
    return logs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    epochs = 30

    set_seed(42)
    train_loader, val_loader = get_data(batch_size=128)

    results = {}

    # ---- D: AdamW (weight_decay=0.01) ----
    set_seed(0)
    model_d = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_d = torch.optim.AdamW(model_d.parameters(), lr=1e-3, weight_decay=0.01)
    results["D_adamw_wd001"] = run_variant(
        "D: AdamW (lr=1e-3, wd=0.01)", model_d, opt_d,
        train_loader, val_loader, device, epochs=epochs
    )

    # ---- E: Adam with eps=0.01 ----
    set_seed(0)
    model_e = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_e = torch.optim.Adam(model_e.parameters(), lr=1e-3, eps=0.01)
    results["E_adam_eps001"] = run_variant(
        "E: Adam (lr=1e-3, eps=0.01)", model_e, opt_e,
        train_loader, val_loader, device, epochs=epochs
    )

    # ---- F: Adam + CosineAnnealingLR ----
    set_seed(0)
    model_f = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_f = torch.optim.Adam(model_f.parameters(), lr=1e-3)
    sched_f = CosineAnnealingLR(opt_f, T_max=epochs, eta_min=0)
    results["F_adam_cosine"] = run_variant(
        "F: Adam (lr=1e-3) + CosineAnnealingLR", model_f, opt_f,
        train_loader, val_loader, device, epochs=epochs, scheduler=sched_f
    )

    # ---- G: AdamW + CosineAnnealingLR (belt and suspenders) ----
    set_seed(0)
    model_g = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_g = torch.optim.AdamW(model_g.parameters(), lr=1e-3, weight_decay=0.01)
    sched_g = CosineAnnealingLR(opt_g, T_max=epochs, eta_min=0)
    results["G_adamw_cosine"] = run_variant(
        "G: AdamW (lr=1e-3, wd=0.01) + CosineAnnealingLR", model_g, opt_g,
        train_loader, val_loader, device, epochs=epochs, scheduler=sched_g
    )

    # Save
    out_dir = Path("outputs/diagnostics/pc_energy_fixes")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "fix_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {out_dir / 'fix_results.json'}")

    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for variant, logs in results.items():
        best_acc = max(l["val_acc"] for l in logs)
        best_ep = next(l["epoch"] for l in logs if l["val_acc"] == best_acc)
        energies = [l["train_energy"] for l in logs]
        init_e, max_e, final_e = energies[0], max(energies), energies[-1]
        exploded = max_e > 10 * init_e if init_e > 0 else False
        print(f"\n  {variant}:")
        print(f"    Best val_acc: {best_acc:.2f}% (epoch {best_ep})")
        print(f"    Energy: {init_e:.4f} → max {max_e:.4f} → final {final_e:.4f}")
        print(f"    Explosion: {'YES ({:.0f}x)'.format(max_e/init_e) if exploded else 'NO'}")


if __name__ == "__main__":
    main()

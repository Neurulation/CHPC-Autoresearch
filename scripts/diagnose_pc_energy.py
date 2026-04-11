"""Diagnostic script: investigate PC-FFNN energy explosion.

Runs the PC-FFNN model on MNIST for N epochs and logs detailed per-layer
diagnostics each epoch to pinpoint the explosion mechanism.

Diagnostics logged:
  - Per-layer prediction error energy: e_l = 0.5 * ||r_{l+1} - f(W_l @ r_l)||²
  - Per-layer weight norms: ||W_l||_F
  - Per-layer gradient norms (PC component vs CE component)
  - Cosine similarity between PC and CE gradients on each shared layer
  - Bottom-up r_{L-1} vs inference-settled r_{L-1} divergence (L2 distance)
  - Validation accuracy and CE loss

Runs 3 variants:
  A) Current (energy + CE through PC layers) — should explode
  B) CE detached from PC layers (stop_grad before cls_head) — tests gradient conflict hypothesis
  C) Current but with SGD (no adaptive LR) — tests Adam amplification hypothesis

Usage:
  python scripts/diagnose_pc_energy.py
"""

import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Add src to path
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
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def per_layer_energy(model, x_flat, reps):
    """Compute energy contribution per layer (with detached reps, active weights)."""
    energies = []
    for i, layer in enumerate(model.layers):
        mu = model._act(i, layer(reps[i].detach()))
        e = reps[i + 1].detach() - mu
        energies.append(0.5 * (e ** 2).sum().item())
    return energies


def compute_gradient_conflict(model, x, y, device):
    """Compute PC energy gradient and CE gradient separately on shared PC layers.

    Returns per-layer: PC grad norm, CE grad norm, cosine similarity.
    """
    B = x.shape[0]
    x_flat = x.view(B, -1)

    # --- PC energy gradient only ---
    model.zero_grad()
    target = F.one_hot(y, model.num_classes).float()
    reps_init = model._bottom_up(x_flat)
    reps_final = model._run_inference(reps_init, clamp_last=target)
    energy = model._energy_active_weights(reps_final)
    energy.backward()

    pc_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            pc_grads[name] = p.grad.clone()

    # --- CE gradient only ---
    model.zero_grad()
    h = x_flat
    for i, layer in enumerate(model.layers[:-1]):
        h = model._act(i, layer(h))
    logits = model.cls_head(h)
    ce_loss = F.cross_entropy(logits, y)
    ce_loss.backward()

    ce_grads = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            ce_grads[name] = p.grad.clone()

    model.zero_grad()

    # --- Compare on shared PC layers ---
    results = {}
    for name in pc_grads:
        if name not in ce_grads:
            continue
        pg = pc_grads[name].flatten()
        cg = ce_grads[name].flatten()
        pc_norm = pg.norm().item()
        ce_norm = cg.norm().item()
        if pc_norm > 0 and ce_norm > 0:
            cos_sim = F.cosine_similarity(pg.unsqueeze(0), cg.unsqueeze(0)).item()
        else:
            cos_sim = 0.0
        results[name] = {
            "pc_grad_norm": pc_norm,
            "ce_grad_norm": ce_norm,
            "cosine_similarity": cos_sim,
        }
    return results, energy.item(), ce_loss.item()


def repr_divergence(model, x_flat, reps_settled):
    """L2 distance between bottom-up r_{L-1} and inference-settled r_{L-1}."""
    # Bottom-up r_{L-1}
    h = x_flat
    with torch.no_grad():
        for i, layer in enumerate(model.layers[:-1]):
            h = model._act(i, layer(h))
    bu_r = h
    settled_r = reps_settled[-2]  # r_{L-1} from inference
    return (bu_r - settled_r).norm().item() / bu_r.shape[0]  # per-sample avg


def train_one_epoch(model, train_loader, optimizer, device, max_grad_norm=None,
                    detach_ce=False):
    """Train one epoch, return avg loss, energy, accuracy."""
    model.train()
    total_loss = 0.0
    total_energy = 0.0
    correct = 0
    total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        if detach_ce:
            # Modified pc_loss: detach before cls_head
            B = x.shape[0]
            x_flat = x.view(B, -1)
            target = F.one_hot(y, model.num_classes).float()
            reps_init = model._bottom_up(x_flat)
            reps_final = model._run_inference(reps_init, clamp_last=target)
            energy = model._energy_active_weights(reps_final)

            # CE with DETACHED bottom-up pass (no grad through PC layers)
            h = x_flat
            with torch.no_grad():
                for i, layer in enumerate(model.layers[:-1]):
                    h = model._act(i, layer(h))
            # h is detached; only cls_head gets CE gradient
            logits = model.cls_head(h)
            ce_loss = F.cross_entropy(logits, y)
            combined_loss = energy + model.ce_weight * ce_loss
        else:
            combined_loss, energy_val, logits = model.pc_loss(x, y)
            energy = torch.tensor(0.0)  # placeholder
            energy_val_scalar = energy_val.item() if hasattr(energy_val, 'item') else energy_val

        combined_loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        total_loss += combined_loss.item()
        if detach_ce:
            total_energy += energy.item()
            _, predicted = torch.max(logits, 1)
        else:
            total_energy += energy_val_scalar
            _, predicted = torch.max(logits, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    n = len(train_loader)
    return {
        "loss": total_loss / n,
        "energy": total_energy / n,
        "accuracy": 100.0 * correct / total,
    }


def validate(model, val_loader, device):
    model.eval()
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = ce_fn(logits, y)
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    n = len(val_loader)
    return {"loss": total_loss / n, "accuracy": 100.0 * correct / total}


def run_diagnostic(variant_name, model, optimizer, train_loader, val_loader,
                   device, epochs=20, max_grad_norm=None, detach_ce=False):
    """Run one full diagnostic experiment."""
    print(f"\n{'='*70}")
    print(f"  VARIANT: {variant_name}")
    print(f"{'='*70}")

    epoch_logs = []
    for epoch in range(1, epochs + 1):
        # --- Train ---
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device,
            max_grad_norm=max_grad_norm, detach_ce=detach_ce
        )

        # --- Validate ---
        val_metrics = validate(model, val_loader, device)

        # --- Per-layer diagnostics (on one batch) ---
        model.eval()
        diag_x, diag_y = next(iter(val_loader))
        diag_x, diag_y = diag_x.to(device), diag_y.to(device)
        B = diag_x.shape[0]
        x_flat = diag_x.view(B, -1)

        # Per-layer energy (with inference-settled reps)
        target = F.one_hot(diag_y, model.num_classes).float()
        reps_init = model._bottom_up(x_flat)
        reps_settled = model._run_inference(reps_init, clamp_last=target)
        layer_energies = per_layer_energy(model, x_flat, reps_settled)

        # Weight norms per layer
        weight_norms = {}
        for i, layer in enumerate(model.layers):
            weight_norms[f"layer_{i}_W"] = layer.weight.norm().item()
            weight_norms[f"layer_{i}_b"] = layer.bias.norm().item()
        weight_norms["cls_head_W"] = model.cls_head.weight.norm().item()

        # Gradient conflict analysis
        model.train()
        grad_conflict, diag_energy, diag_ce = compute_gradient_conflict(
            model, diag_x, diag_y, device
        )

        # Representation divergence
        model.eval()
        reps_free = model._run_inference(reps_init, clamp_last=None)
        bu_vs_settled = repr_divergence(model, x_flat, reps_settled)
        free_vs_settled_rLm1 = (reps_free[-2] - reps_settled[-2]).norm().item() / B

        # Summarize gradient conflict per PC layer
        layer_conflict = {}
        for name, info in grad_conflict.items():
            if "layers" in name and "weight" in name:
                layer_conflict[name] = info

        log_entry = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 4),
            "train_energy": round(train_metrics["energy"], 4),
            "train_acc": round(train_metrics["accuracy"], 2),
            "val_loss": round(val_metrics["loss"], 4),
            "val_acc": round(val_metrics["accuracy"], 2),
            "per_layer_energy": [round(e, 4) for e in layer_energies],
            "weight_norms": {k: round(v, 4) for k, v in weight_norms.items()},
            "grad_conflict": {
                k: {kk: round(vv, 6) for kk, vv in v.items()}
                for k, v in layer_conflict.items()
            },
            "repr_divergence": {
                "bottomup_vs_settled_rLm1": round(bu_vs_settled, 4),
                "free_vs_clamped_settled_rLm1": round(free_vs_settled_rLm1, 4),
            },
            "diag_batch_energy": round(diag_energy, 4),
            "diag_batch_ce": round(diag_ce, 4),
        }
        epoch_logs.append(log_entry)

        # Print summary
        layer_e_str = " | ".join(f"L{i}={e:.1f}" for i, e in enumerate(layer_energies))
        conflict_str = ""
        for name, info in layer_conflict.items():
            cos = info["cosine_similarity"]
            conflict_str += f"  {name}: cos={cos:+.4f} (PC={info['pc_grad_norm']:.4f}, CE={info['ce_grad_norm']:.4f})\n"

        print(f"\nEpoch {epoch:2d} | Train E={train_metrics['energy']:.4f} Acc={train_metrics['accuracy']:.1f}% | "
              f"Val Acc={val_metrics['accuracy']:.2f}% CE={val_metrics['loss']:.4f}")
        print(f"  Per-layer energy: {layer_e_str}")
        print(f"  Repr divergence: BU↔settled={bu_vs_settled:.4f}  free↔clamped={free_vs_settled_rLm1:.4f}")
        if conflict_str:
            print(f"  Gradient conflict:\n{conflict_str}", end="")

    return epoch_logs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    set_seed(42)
    train_loader, val_loader = get_data(batch_size=128, val_split=0.1)
    epochs = 20

    results = {}

    # ---- VARIANT A: Current (energy + CE through PC layers) ----
    set_seed(0)
    model_a = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=1e-3)
    results["A_current"] = run_diagnostic(
        "A: Current (energy + CE→PC layers, Adam, no clip)",
        model_a, opt_a, train_loader, val_loader, device,
        epochs=epochs, max_grad_norm=None, detach_ce=False
    )

    # ---- VARIANT B: CE detached from PC layers ----
    set_seed(0)
    model_b = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=1e-3)
    results["B_ce_detached"] = run_diagnostic(
        "B: CE detached from PC layers (only cls_head gets CE grad)",
        model_b, opt_b, train_loader, val_loader, device,
        epochs=epochs, max_grad_norm=None, detach_ce=True
    )

    # ---- VARIANT C: Current but with SGD ----
    set_seed(0)
    model_c = PCFFNN(input_size=784, hidden_dims=[256, 128], num_classes=10,
                     T_pc=20, lr_pc=0.05, ce_weight=1.0).to(device)
    opt_c = torch.optim.SGD(model_c.parameters(), lr=1e-3, momentum=0.9)
    results["C_sgd"] = run_diagnostic(
        "C: Current (energy + CE→PC layers, SGD+mom, no clip)",
        model_c, opt_c, train_loader, val_loader, device,
        epochs=epochs, max_grad_norm=None, detach_ce=False
    )

    # Save results
    out_dir = Path("outputs/diagnostics/pc_energy_explosion")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {out_dir / 'diagnostic_results.json'}")

    # ---- Summary ----
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for variant, logs in results.items():
        best_acc = max(l["val_acc"] for l in logs)
        best_epoch = next(l["epoch"] for l in logs if l["val_acc"] == best_acc)
        final_energy = logs[-1]["train_energy"]
        max_energy = max(l["train_energy"] for l in logs)
        spike_epoch = next(l["epoch"] for l in logs if l["train_energy"] == max_energy)

        # Check if energy exploded (>10x the initial energy)
        init_energy = logs[0]["train_energy"]
        exploded = max_energy > 10 * init_energy if init_energy > 0 else False

        print(f"\n  {variant}:")
        print(f"    Best val_acc: {best_acc:.2f}% (epoch {best_epoch})")
        print(f"    Energy: init={init_energy:.4f} → max={max_energy:.4f} (epoch {spike_epoch}) → final={final_energy:.4f}")
        print(f"    Explosion: {'YES (>{0:.0f}x)'.format(max_energy/init_energy) if exploded else 'NO'}")


if __name__ == "__main__":
    main()

"""Spiking Predictive Coding Feed-Forward Neural Network (SPC-FFNN).

Combines PC inference with LIF spiking dynamics for initial representation
generation. The SNN provides bottom-up spike-count representations via rate
coding; the PC inference loop then refines these in the continuous rate domain.

Two timescales are kept strictly separate:
  1. SNN temporal loop (T=25 Bernoulli timesteps): generates spike-count reps.
  2. PC inference loop (T_pc gradient steps): refines reps after SNN.
No interleaving → no meta-gradient interference (cf. PC-RNN failure, iter 5).

Architecture
------------
Input    : (N, 1, 28, 28) MNIST → flatten → (N, 784)
Rate enc : (T, N, 784) Bernoulli spikes
SNN FFNN : Linear(784→256)+LIF → spike counts r₁ ∈ ℝ^256
           Linear(256→128)+LIF → spike counts r₂ ∈ ℝ^128
SNN out  : Linear(128→10)      → spike counts r₃ ∈ ℝ^10
PC inf   : T_pc gradient steps on {r₁, r₂} (r₃ clamped = one_hot(y) in training)
Weight Δ : ∂F/∂W via active-weight energy (same as PC-FFNN)
Eval     : cls_head(r₂) after free inference (no clamping)

Total parameters: ~537,610 (PC-FFNN v3 has 506,010 — comparable).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate


class SPCFFNNModel(nn.Module):
    """Spiking Predictive Coding FFNN for MNIST classification.

    Args:
        input_size:  Flattened input dimension (784 for MNIST).
        hidden_dims: Hidden layer widths, e.g. [256, 128].
        num_classes: Number of output classes.
        T_pc:        PC inference steps per sample (more → closer to FE minimum).
        lr_pc:       Step size for representation updates during PC inference.
        ce_weight:   Cross-entropy loss weight relative to PC energy.
        beta:        LIF membrane decay constant (0 < beta < 1).
        threshold:   LIF firing threshold.
        timesteps:   SNN rate-coding timesteps T.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        T_pc: int = 20,
        lr_pc: float = 0.05,
        ce_weight: float = 1.0,
        beta: float = 0.9,
        threshold: float = 1.0,
        timesteps: int = 25,
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight
        self.timesteps = timesteps
        self._n_hidden = len(hidden_dims)

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Weight matrices: same as PC-FFNN
        dims = [input_size] + hidden_dims + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # LIF neurons — one per layer (shared across timesteps)
        self.lif_neurons = nn.ModuleList(
            [
                snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
                for _ in range(len(self.layers))
            ]
        )

        # Classification head (training-eval mismatch fix, same as PC-FFNN v3)
        self.cls_head = nn.Linear(hidden_dims[-1], num_classes)

    # ------------------------------------------------------------------
    # Activation (used in PC energy computation)
    # ------------------------------------------------------------------

    def _act(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """ReLU for hidden layers, identity for output — same as PC-FFNN."""
        if layer_idx < self._n_hidden:
            return F.relu(x)
        return x

    # ------------------------------------------------------------------
    # SNN bottom-up pass
    # ------------------------------------------------------------------

    def _bottom_up(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """SNN bottom-up pass: rate-encode input, run LIF layers, return spike counts.

        Args:
            x_flat: (N, input_size) flattened images in [0, 1].

        Returns:
            [r_0=x_flat, r_1, ..., r_L] — all detached spike-count representations.
            r_0: raw pixel inputs (used as PC layer-0 representation).
            r_1..r_L: accumulated spike counts from each LIF layer.
        """
        x_flat = x_flat.clamp(0.0, 1.0)

        # Rate-encode input: (T, N, input_size) Bernoulli spike trains
        spikes_in = torch.bernoulli(
            x_flat.unsqueeze(0).expand(self.timesteps, -1, -1)
        )

        # Run SNN forward
        mems = [lif.init_leaky() for lif in self.lif_neurons]
        spike_accs = [torch.zeros(x_flat.shape[0], dim, device=x_flat.device)
                      for dim in list(self.hidden_dims) + [self.num_classes]]

        h = spikes_in  # (T, N, input_size) at start; updated per layer

        # For each layer, accumulate spikes over T timesteps
        current_input = spikes_in  # (T, N, layer_input_dim)
        for l_idx, (layer, lif) in enumerate(zip(self.layers, self.lif_neurons)):
            mem = mems[l_idx]
            spk_acc = spike_accs[l_idx]
            spk_rec = []
            for t in range(self.timesteps):
                cur = layer(current_input[t])          # (N, layer_output_dim)
                spk, mem = lif(cur, mem)               # LIF step
                spk_acc = spk_acc + spk
                spk_rec.append(spk)

            spike_accs[l_idx] = spk_acc.detach()
            current_input = torch.stack(spk_rec)  # (T, N, layer_output_dim)

        # r_0 = raw pixel input (detached), r_1..r_L = normalised firing rates in [0,1]
        reps = [x_flat.detach()] + [acc.detach() / self.timesteps for acc in spike_accs]
        return reps

    # ------------------------------------------------------------------
    # PC inference helpers (identical to PC-FFNN)
    # ------------------------------------------------------------------

    def _energy_fixed_weights(
        self, r0: torch.Tensor, r_mutable: List[torch.Tensor]
    ) -> torch.Tensor:
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable
        for i, layer in enumerate(self.layers):
            w = layer.weight.detach()
            b = layer.bias.detach()
            mu = self._act(i, F.linear(all_r[i], w, b))
            e = all_r[i + 1] - mu
            energy = energy + 0.5 * (e ** 2).sum()
        return energy

    def _energy_active_weights(self, reps: List[torch.Tensor]) -> torch.Tensor:
        energy = reps[0].new_zeros(())
        for i, layer in enumerate(self.layers):
            mu = self._act(i, layer(reps[i].detach()))
            e = reps[i + 1].detach() - mu
            energy = energy + 0.5 * (e ** 2).sum()
        return energy

    def _run_inference(
        self,
        reps: List[torch.Tensor],
        clamp_last: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        r0 = reps[0].detach()
        r_free = [r.detach().clone() for r in reps[1:]]

        if clamp_last is not None:
            r_free[-1] = clamp_last.detach().float()

        with torch.enable_grad():
            for r in r_free:
                r.requires_grad_(True)

            for _ in range(self.T_pc):
                update_reps = r_free[:-1] if clamp_last is not None else r_free
                if not update_reps:
                    break
                energy = self._energy_fixed_weights(r0, r_free)
                grads = torch.autograd.grad(energy, update_reps, create_graph=False)
                updated = [
                    (r - self.lr_pc * g).detach().requires_grad_(True)
                    for r, g in zip(update_reps, grads)
                ]
                r_free = (updated + [r_free[-1]]) if clamp_last is not None else updated

        return [r0] + [r.detach() for r in r_free]

    # ------------------------------------------------------------------
    # Public API (matches PC-FFNN / train_pcnn.py interface)
    # ------------------------------------------------------------------

    def pc_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PC training: SNN bottom-up pass → supervised inference → weight-update energy.

        Args:
            x: Images (N, C, H, W).
            y: Labels (N,).

        Returns:
            combined_loss: energy + ce_weight * ce_loss (backprop this).
            energy:        PC free energy (detached, for logging).
            logits:        cls_head(r_{L-1}) logits (detached, for accuracy logging).
        """
        x_flat = x.view(x.shape[0], -1)

        # Phase 1 — Inference (clamp r_L = one_hot(y))
        target = F.one_hot(y, self.num_classes).float()
        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_inference(reps_init, clamp_last=target)

        # Phase 2 — Weight-update energy
        energy = self._energy_active_weights(reps_final)

        # CE head on r_{L-1}: re-run feedforward (differentiable) for gradients
        h = x_flat
        for i, layer in enumerate(self.layers[:-1]):
            h = self._act(i, layer(h))
        logits = self.cls_head(h)
        ce_loss = F.cross_entropy(logits, y)

        return energy + self.ce_weight * ce_loss, energy.detach(), logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation: SNN bottom-up → free inference → cls_head(r_{L-1}).

        Args:
            x: Images (N, C, H, W).

        Returns:
            logits: (N, num_classes).
        """
        x_flat = x.view(x.shape[0], -1)
        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_inference(reps_init, clamp_last=None)
        return self.cls_head(reps_final[-2])  # r_{L-1}

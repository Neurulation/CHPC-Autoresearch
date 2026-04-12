"""Spiking Predictive Coding Feed-Forward Neural Network (SPC-FFNN).

Combines PC inference with LIF spiking dynamics for initial representation
generation. The SNN provides bottom-up spike-count representations via rate
coding; the PC inference loop then refines these in the continuous rate domain.

Three gradient modes (ce_mode):

  v2 (baseline, iter 3):
    SNN outputs are fully detached. CE only trains cls_head. self.layers
    receive gradient from PC energy only (reconstruction objective). SNN and
    PC are coupled only at the representation level, not via gradients.
    Result: 55.95% — PC energy alone is insufficient for discrimination.

  a (Option A, iter 4):
    SNN spike accumulators are NOT detached. CE backpropagates through the
    SNN via surrogate gradients (fast_sigmoid) all the way into self.layers.
    self.layers receive gradient from BOTH CE (discriminative) and PC energy
    (reconstructive). The two signals are additive under Adam.
    Hypothesis: CE dominates and drives discriminative features.

  d (Option D, iter 5):
    SNN outputs remain detached (no SNN meta-gradient). CE is computed on
    the FREE-INFERENCE PC equilibrium output (reps_free[-1] after T_pc steps
    with no label clamping) rather than on raw SNN spike counts. CE gradient
    backpropagates through T_pc inference steps (BPTT through the inner
    optimisation loop) back into self.layers via the energy function.
    self.layers receive gradient from BOTH: PC weight-update energy (clamped
    inference) and CE-through-equilibrium (free differentiable inference).
    Hypothesis: the PC equilibrium is a more principled classification target,
    and BPTT-through-inference gives cleaner gradient signal to self.layers.
    Cost: 2× inference compute + create_graph overhead (use T_pc=10 in cfg).

Architecture
------------
Input    : (N, 1, 28, 28) MNIST → flatten → (N, 784)
Rate enc : (T, N, 784) Bernoulli spikes
SNN FFNN : Linear(784→256)+LIF → spike counts r₁ ∈ ℝ^256
           Linear(256→128)+LIF → spike counts r₂ ∈ ℝ^128
SNN out  : Linear(128→10)      → spike counts r₃ ∈ ℝ^10
PC inf   : T_pc gradient steps on {r₁, r₂} (r₃ clamped = one_hot(y) in training)
Weight Δ : ∂F/∂W via active-weight energy (same as PC-FFNN)
Eval     : cls_head(r₂) [modes v2/a] or reps_free[-1] [mode d]
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
        T_pc:        PC inference steps per sample.
        lr_pc:       Step size for representation updates during PC inference.
        ce_weight:   Cross-entropy loss weight relative to PC energy.
        beta:        LIF membrane decay constant (0 < beta < 1).
        threshold:   LIF firing threshold.
        timesteps:   SNN rate-coding timesteps T.
        ce_mode:     Gradient routing mode: 'v2' | 'a' | 'd' (see module docstring).
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
        ce_mode: str = "v2",
    ) -> None:
        super().__init__()

        assert ce_mode in ("v2", "a", "d"), f"ce_mode must be 'v2', 'a', or 'd', got {ce_mode!r}"

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight
        self.timesteps = timesteps
        self.ce_mode = ce_mode
        self._n_hidden = len(hidden_dims)

        spike_grad = surrogate.fast_sigmoid(slope=25)

        dims = [input_size] + hidden_dims + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        self.lif_neurons = nn.ModuleList(
            [
                snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
                for _ in range(len(self.layers))
            ]
        )

        # cls_head used by modes v2 and a only.
        # Mode d classifies directly from the PC free-inference output.
        self.cls_head = nn.Linear(hidden_dims[-1], num_classes)

    # ------------------------------------------------------------------
    # Activation (used in PC energy computation)
    # ------------------------------------------------------------------

    def _act(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """ReLU for hidden layers, identity for output."""
        if layer_idx < self._n_hidden:
            return F.relu(x)
        return x

    # ------------------------------------------------------------------
    # SNN bottom-up pass
    # ------------------------------------------------------------------

    def _bottom_up(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """SNN bottom-up pass: rate-encode input, run LIF layers, return spike counts.

        Mode v2/d: spike accumulators are detached — SNN is isolated from
                   the PC and CE gradient graphs.
        Mode a:    spike accumulators retain their computation graph so that
                   CE gradient can backpropagate through LIF surrogate grads
                   into self.layers.

        Returns:
            [r_0=x_flat, r_1, ..., r_L] spike-count representations.
        """
        x_flat = x_flat.clamp(0.0, 1.0)

        spikes_in = torch.bernoulli(
            x_flat.unsqueeze(0).expand(self.timesteps, -1, -1)
        )

        mems = [lif.init_leaky() for lif in self.lif_neurons]
        spike_accs = [
            torch.zeros(x_flat.shape[0], dim, device=x_flat.device)
            for dim in list(self.hidden_dims) + [self.num_classes]
        ]

        current_input = spikes_in
        for l_idx, (layer, lif) in enumerate(zip(self.layers, self.lif_neurons)):
            mem = mems[l_idx]
            spk_acc = spike_accs[l_idx]
            spk_rec = []
            for t in range(self.timesteps):
                cur = layer(current_input[t])
                spk, mem = lif(cur, mem)
                spk_acc = spk_acc + spk
                spk_rec.append(spk)

            # Mode 'a': keep graph so CE flows through SNN via surrogate grads.
            # Modes 'v2'/'d': detach — SNN isolated from backward pass.
            if self.ce_mode == "a":
                spike_accs[l_idx] = spk_acc
            else:
                spike_accs[l_idx] = spk_acc.detach()

            current_input = torch.stack(spk_rec)

        r0 = x_flat.detach()
        if self.ce_mode == "a":
            reps = [r0] + [acc / self.timesteps for acc in spike_accs]
        else:
            reps = [r0] + [acc.detach() / self.timesteps for acc in spike_accs]
        return reps

    # ------------------------------------------------------------------
    # PC inference helpers
    # ------------------------------------------------------------------

    def _energy_fixed_weights(
        self, r0: torch.Tensor, r_mutable: List[torch.Tensor]
    ) -> torch.Tensor:
        """PC free energy with DETACHED weights — for representation update only."""
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable
        for i, layer in enumerate(self.layers):
            w = layer.weight.detach()
            b = layer.bias.detach()
            mu = self._act(i, F.linear(all_r[i], w, b))
            e = all_r[i + 1] - mu
            energy = energy + 0.5 * (e ** 2).sum()
        return energy

    def _energy_fixed_weights_differentiable(
        self, r0: torch.Tensor, r_mutable: List[torch.Tensor]
    ) -> torch.Tensor:
        """PC free energy with LIVE weights — for BPTT through inference (mode d).

        Unlike _energy_fixed_weights, weights are NOT detached, so backward
        through this energy (via create_graph=True autograd.grad) propagates
        gradient into self.layers.
        """
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable
        for i, layer in enumerate(self.layers):
            mu = self._act(i, layer(all_r[i]))
            e = all_r[i + 1] - mu
            energy = energy + 0.5 * (e ** 2).sum()
        return energy

    def _energy_active_weights(self, reps: List[torch.Tensor]) -> torch.Tensor:
        """PC weight-update energy: gradient flows through weights, reps detached."""
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
        """Standard (non-differentiable) PC inference — fast, detaches between steps."""
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

    def _run_inference_differentiable(
        self,
        reps: List[torch.Tensor],
        clamp_last: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Differentiable PC inference for BPTT through inference steps (mode d).

        Keeps the computation graph alive between steps (no .detach()) and uses
        _energy_fixed_weights_differentiable (weights NOT detached). CE gradient
        on the output of this function backpropagates through all T_pc steps into
        self.layers.

        The INPUT reps must already be detached before calling (SNN stays isolated
        from BPTT — no SNN meta-gradient). create_graph=True is used to enable
        this chain.
        """
        r0 = reps[0].detach()
        # Start from SNN reps — detached at source, but graph is live after first step
        r_free = [r.detach().clone().requires_grad_(True) for r in reps[1:]]

        if clamp_last is not None:
            r_free[-1] = clamp_last.detach().float()

        for _ in range(self.T_pc):
            update_reps = r_free[:-1] if clamp_last is not None else r_free
            if not update_reps:
                break
            energy = self._energy_fixed_weights_differentiable(r0, r_free)
            # retain_graph=True: each step's forward graph must stay alive because
            # later steps' graphs chain through it (r_{k+1} depends on grads_k
            # which depends on energy_k's graph). Released after combined_loss.backward().
            grads = torch.autograd.grad(
                energy,
                update_reps,
                create_graph=True,
                retain_graph=True,
            )
            # No .detach() — computation graph kept alive for BPTT
            updated = [r - self.lr_pc * g for r, g in zip(update_reps, grads)]
            r_free = (updated + [r_free[-1]]) if clamp_last is not None else updated

        return [r0] + r_free

    # ------------------------------------------------------------------
    # Public API (matches PC-FFNN / train_pcnn.py interface)
    # ------------------------------------------------------------------

    def pc_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """PC training loss, branching on ce_mode.

        All modes:
          - Clamped PC inference → PC weight-update energy (trains self.layers via
            reconstruction, unchanged across all modes).

        Mode v2:
          - CE on cls_head(reps_init[-2]): only trains cls_head. self.layers
            receive no CE signal (detached SNN).

        Mode a:
          - CE on cls_head(reps_init[-2]): SNN spike accumulators are NOT detached,
            so CE backpropagates through LIF surrogate grads into self.layers.
            self.layers trained by CE (discriminative) + PC energy (reconstructive).

        Mode d:
          - Free differentiable inference → CE on reps_free[-1] (PC output
            equilibrium). BPTT through T_pc inference steps propagates CE
            gradient into self.layers via the energy function. SNN stays detached.
            Two inference passes per step: clamped (PC energy) + free (CE).

        Returns:
            combined_loss, energy (detached, for logging), logits (detached).
        """
        x_flat = x.view(x.shape[0], -1)
        target = F.one_hot(y, self.num_classes).float()

        reps_init = self._bottom_up(x_flat)

        # PC weight-update energy (same for all modes)
        reps_final = self._run_inference(reps_init, clamp_last=target)
        energy = self._energy_active_weights(reps_final)

        if self.ce_mode == "d":
            # CE via BPTT through free inference equilibrium
            reps_free = self._run_inference_differentiable(reps_init)
            logits = reps_free[-1]  # PC output representation (N, num_classes)
            ce_loss = F.cross_entropy(logits, y)
        else:
            # Modes v2 and a: CE on cls_head applied to SNN representation r_{L-1}
            # Mode a: reps_init[-2] has gradient graph (SNN not detached)
            # Mode v2: reps_init[-2] is detached (no CE signal to self.layers)
            logits = self.cls_head(reps_init[-2])
            ce_loss = F.cross_entropy(logits, y)

        return energy + self.ce_weight * ce_loss, energy.detach(), logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation forward pass.

        Mode v2/a: cls_head(SNN spike counts r_{L-1}) — pure feedforward, no PC.
        Mode d:    non-differentiable free PC inference → reps_free[-1].
                   Uses _run_inference (fast, detached) since no backward needed.
        """
        x_flat = x.view(x.shape[0], -1)
        reps_init = self._bottom_up(x_flat)
        if self.ce_mode == "d":
            reps_free = self._run_inference(reps_init)
            return reps_free[-1]
        return self.cls_head(reps_init[-2])

"""Temporal Predictive Coding RNN (PC-RNN).

Extends PC-FFNN (Rao & Ballard 1999) with recurrent connections.

At each timestep t, each hidden layer l has:
    Prediction: mu_l(t) = f(W_ff[l-1] @ r_{l-1}(t) + W_rec[l-1] @ r_l(t-1))
    Error:      e_l(t) = r_l(t) - mu_l(t)

The output layer uses feedforward-only prediction (no recurrence):
    mu_L(t) = f(W_ff[-1] @ r_{L-1}(t))

PC free energy (per timestep):
    F_t = 0.5 * Σ_l ||r_l(t) - mu_l(t)||²

Total energy over sequence:
    F = Σ_t F_t

Training uses the same two-phase approach as PC-FFNN:
    Phase 1 — Inference: greedy per-timestep gradient descent on r_l(t)
                         with weights fixed. Hidden state h_l = r_l(t) carries
                         to the next timestep as the recurrent context.
                         At the final timestep: r_L is clamped to one_hot(y).
    Phase 2 — Weight update: recompute F over the full sequence with weights
                               active (settled representations detached).
                               Combined with CE on cls_head(h_T) via BPTT.

Architecture (sequential MNIST default):
    T=28 timesteps (one 28-pixel row per step)
    input_size=28, hidden_dims=[256, 128], num_classes=10

References:
  - Rao & Ballard (1999). Predictive coding in the visual cortex.
  - Elman (1990). Finding structure in time (SRN formulation extended to PC).
  - Friston (2005). A theory of cortical responses (predictive coding / free energy).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCRNN(nn.Module):
    """Temporal Predictive Coding RNN for sequence classification.

    Args:
        input_size: Feature dimension per timestep (28 for MNIST rows)
        hidden_dims: Hidden layer widths, e.g. [256, 128]
        num_classes: Number of output classes
        T_pc: Inference steps per timestep (fewer OK due to warm-start from h_prev)
        lr_pc: Step size for representation updates during inference
        ce_weight: Weight for cross-entropy loss relative to PC energy
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        T_pc: int = 10,
        lr_pc: float = 0.05,
        ce_weight: float = 1.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight

        dims = [input_size] + list(hidden_dims) + [num_classes]
        self._n_hidden = len(hidden_dims)
        self._n_layers = len(dims) - 1

        # Feedforward PC weights: layers[i] maps dims[i] → dims[i+1]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(self._n_layers)]
        )

        # Recurrent weights: W_rec[i] maps dims[i+1] → dims[i+1]
        # Only hidden layers get recurrent connections (index i < n_hidden)
        self.W_rec = nn.ModuleList(
            [nn.Linear(dims[i + 1], dims[i + 1], bias=False) for i in range(self._n_hidden)]
        )

        # CE classification head trained on the final hidden rep at the last timestep
        self.cls_head = nn.Linear(hidden_dims[-1], num_classes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _act(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """ReLU for hidden layers, identity for output layer."""
        if layer_idx < self._n_hidden:
            return F.relu(x)
        return x

    def _predict_detached(
        self,
        layer_idx: int,
        r_below: torch.Tensor,
        h_prev_l: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute prediction mu for layer (layer_idx+1) with weights detached.

        Used during inference phase — gradients flow only to representations.
        h_prev_l is always detached to prevent BPTT through time.
        """
        layer = self.layers[layer_idx]
        ff = F.linear(
            r_below,
            layer.weight.detach(),
            layer.bias.detach() if layer.bias is not None else None,
        )
        if layer_idx < self._n_hidden and h_prev_l is not None:
            w_rec = self.W_rec[layer_idx]
            ff = ff + F.linear(h_prev_l.detach(), w_rec.weight.detach())
        return self._act(layer_idx, ff)

    def _energy_fixed_weights_t(
        self,
        r0: torch.Tensor,
        r_mutable: List[torch.Tensor],
        h_prev: List[torch.Tensor],
    ) -> torch.Tensor:
        """PC energy at a single timestep with weights detached.

        Gradients flow only to r_mutable (used during the inference phase).
        h_prev is always detached to prevent BPTT.

        Args:
            r0: Clamped input r[0] (detached)
            r_mutable: [r[1], ..., r[n_layers]] — may require grad
            h_prev: [h_prev[0], ..., h_prev[n_hidden-1]] — previous hidden states
        """
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable
        for i in range(self._n_layers):
            h_p = h_prev[i] if i < self._n_hidden else None
            mu = self._predict_detached(i, all_r[i], h_p)
            e = all_r[i + 1] - mu
            energy = energy + 0.5 * (e ** 2).sum()
        return energy

    def _energy_active_weights_seq(
        self,
        all_reps: List[List[torch.Tensor]],
    ) -> torch.Tensor:
        """Recompute full-sequence energy with weights active.

        Uses settled representations from inference (detached).
        Gradients flow only to self.layers and self.W_rec.

        Args:
            all_reps: all_reps[t][l] = detached settled tensor, l=0..n_layers
        """
        T = len(all_reps)
        B = all_reps[0][0].shape[0]
        dims = [self.input_size] + self.hidden_dims + [self.num_classes]

        h_prev = [
            torch.zeros(B, dims[i + 1], device=all_reps[0][0].device)
            for i in range(self._n_hidden)
        ]
        energy = all_reps[0][0].new_zeros(())

        for t in range(T):
            reps_t = all_reps[t]
            for i, layer in enumerate(self.layers):
                r_below = reps_t[i].detach()
                r_target = reps_t[i + 1].detach()

                ff = layer(r_below)
                if i < self._n_hidden:
                    ff = ff + self.W_rec[i](h_prev[i].detach())
                mu = self._act(i, ff)
                e = r_target - mu
                energy = energy + 0.5 * (e ** 2).mean()

            # Advance h_prev from settled hidden states
            for i in range(self._n_hidden):
                h_prev[i] = reps_t[i + 1].detach().clone()

        return energy

    def _run_inference_seq(
        self,
        x_seq: torch.Tensor,
        y: Optional[torch.Tensor] = None,
    ) -> Tuple[List[List[torch.Tensor]], float]:
        """Greedy per-timestep PC inference over the full sequence.

        At each timestep t:
          - Warm-start representations from previous settled hidden states.
          - Run T_pc gradient steps on hidden reps (and output if not clamped).
          - Pass settled hidden states to the next timestep.

        Args:
            x_seq: (B, T, input_size)
            y: (B,) class labels — if given, clamps r_L = one_hot(y) at t=T-1

        Returns:
            all_reps: all_reps[t][l] = detached settled tensor, l=0..n_layers
            energy_log: float — summed energy for W&B logging
        """
        B, T, _ = x_seq.shape
        dims = [self.input_size] + self.hidden_dims + [self.num_classes]

        # h_prev[i] = previous settled r[i+1] for the recurrent term
        h_prev = [
            torch.zeros(B, dims[i + 1], device=x_seq.device)
            for i in range(self._n_hidden)
        ]

        all_reps: List[List[torch.Tensor]] = []
        total_energy_log = 0.0

        with torch.enable_grad():
            for t in range(T):
                x_t = x_seq[:, t, :].detach()
                is_last = t == T - 1

                # Warm-start hidden reps from previous settled states
                r_mutable = [h.detach().clone() for h in h_prev]

                # Output rep: clamped to one_hot(y) at last training step
                if y is not None and is_last:
                    r_out = F.one_hot(y, self.num_classes).float().to(x_seq.device)
                else:
                    with torch.no_grad():
                        r_out = self._predict_detached(
                            self._n_layers - 1, r_mutable[-1], None
                        )
                r_mutable.append(r_out.detach().clone())

                # Enable gradients on free representations
                for r in r_mutable:
                    r.requires_grad_(True)

                # T_pc inference steps
                for _ in range(self.T_pc):
                    # Hidden layers always free; output clamped at last training step
                    if y is not None and is_last:
                        update_reps = r_mutable[:-1]  # r[1..n_hidden] only
                    else:
                        update_reps = r_mutable        # r[1..n_layers]

                    if not update_reps:
                        break

                    energy = self._energy_fixed_weights_t(x_t, r_mutable, h_prev)
                    grads = torch.autograd.grad(energy, update_reps, create_graph=False)

                    updated = [
                        (r - self.lr_pc * g).detach().requires_grad_(True)
                        for r, g in zip(update_reps, grads)
                    ]
                    if y is not None and is_last:
                        r_mutable = updated + [r_mutable[-1]]
                    else:
                        r_mutable = updated

                # Log energy at the settled state
                with torch.no_grad():
                    all_r = [x_t] + [r.detach() for r in r_mutable]
                    for i in range(self._n_layers):
                        h_p = h_prev[i] if i < self._n_hidden else None
                        mu = self._predict_detached(i, all_r[i], h_p)
                        e = all_r[i + 1] - mu
                        total_energy_log += 0.5 * (e ** 2).mean().item()

                # Store settled reps, advance hidden states
                settled = [x_t.detach()] + [r.detach() for r in r_mutable]
                all_reps.append(settled)
                for i in range(self._n_hidden):
                    h_prev[i] = settled[i + 1].clone()

        return all_reps, total_energy_log

    # ------------------------------------------------------------------
    # Public API (matches PC-FFNN interface for compatibility with train_pcnn.py)
    # ------------------------------------------------------------------

    def pc_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PC-RNN training loss.

        Two-phase approach (mirrors PC-FFNN):
          1. Inference: settle representations per timestep with weights fixed.
          2. Weight update: PC energy (weights active, reps detached)
                          + CE loss via standard BPTT forward pass.

        The CE BPTT forward gives gradient signals to both W_ff and W_rec,
        complementing the PC energy's local update signal.

        Args:
            x: Images (B, C, H, W) or (B, T, input_size)
            y: Class labels (B,)

        Returns:
            combined_loss: energy + ce_weight * ce_loss — backprop this.
            energy: PC free energy scalar (for W&B logging).
            logits: (B, num_classes) from cls_head(h_T) — for accuracy logging.
        """
        B = x.shape[0]
        x_seq = x.squeeze(1) if x.dim() == 4 else x  # (B, T, input_size)

        # Phase 1 — Inference (weights fixed, reps updated)
        all_reps, _ = self._run_inference_seq(x_seq, y=y)

        # Phase 2 — PC energy with active weights (reps detached)
        energy = self._energy_active_weights_seq(all_reps)

        # CE head via BPTT: full sequence forward through W_ff + W_rec
        # Gives CE gradients to both feedforward and recurrent weights
        h = [torch.zeros(B, d, device=x.device) for d in self.hidden_dims]
        for t in range(x_seq.shape[1]):
            x_t = x_seq[:, t, :]
            r = x_t
            new_h = []
            for i, (layer, w_rec) in enumerate(zip(self.layers[:-1], self.W_rec)):
                r = self._act(i, layer(r) + w_rec(h[i]))
                new_h.append(r)
            h = new_h
        logits = self.cls_head(r)
        ce_loss = F.cross_entropy(logits, y)

        combined_loss = energy + self.ce_weight * ce_loss
        return combined_loss, energy.detach(), logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation: free PC inference, then cls_head on final hidden state.

        Args:
            x: Images (B, C, H, W) or (B, T, input_size)

        Returns:
            logits: (B, num_classes)
        """
        x_seq = x.squeeze(1) if x.dim() == 4 else x
        all_reps, _ = self._run_inference_seq(x_seq, y=None)
        h_last = all_reps[-1][self._n_hidden]  # settled r[n_hidden] at final timestep
        return self.cls_head(h_last)

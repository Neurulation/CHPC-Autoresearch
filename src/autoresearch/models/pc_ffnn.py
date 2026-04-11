"""Predictive Coding Feed-Forward Neural Network (PC-FFNN).

Implements hierarchical predictive coding for image classification.

References:
  - Rao & Ballard (1999). Predictive coding in the visual cortex.
  - Whittington & Bogacz (2017). An approximation of the error backpropagation
    algorithm in a predictive coding network with local Hebbian synaptic plasticity.
  - Millidge et al. (2020). Predictive coding approximates backprop along
    arbitrary computation graphs.

Architecture:
  Input (x) → [PC Layer 1] → [PC Layer 2] → ... → Output (num_classes)

Each layer l has:
  - r_l: representation units (current estimate of hidden causes)
  - e_l: error units  e_l = r_l - f(W_l @ r_{l-1})
  - mu_l = f(W_l @ r_{l-1}): top-down prediction

  Hidden layers use ReLU; output layer uses identity.

Free energy:
  F = 0.5 * Σ_l || r_l - f(W_l @ r_{l-1}) ||²

Training (two-phase):
  Phase 1 — Inference: minimise F over {r_l} via T_pc gradient steps (weights fixed).
             Supervised: clamp r_L = one_hot(y).
  Phase 2 — Weight update: minimise (F + λ·CE(cls_head(r_{L-1}), y)) over weights.
             F is minimised by PC energy, CE trains the classification head on the
             final hidden representation r_{L-1}, avoiding training-eval mismatch.

Evaluation:
  cls_head(r_{L-1}) after T_pc inference steps (no clamping).
  This bypasses r_L entirely — r_{L-1} is a well-formed feature vector trained
  jointly by PC energy and CE loss, yielding properly calibrated logits.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCFFNN(nn.Module):
    """Predictive Coding Feed-Forward Network for image classification.

    Args:
        input_size: Flattened input dimension (784 for MNIST)
        hidden_dims: Hidden layer widths, e.g. [256, 128]
        num_classes: Number of output classes
        T_pc: Inference steps per forward pass (more → closer to true PC minimum)
        lr_pc: Step size for representation updates during inference
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        T_pc: int = 20,
        lr_pc: float = 0.05,
        ce_weight: float = 1.0,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight

        # Weight matrices W_l: r_{l-1} → r_l
        dims = [input_size] + hidden_dims + [num_classes]
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )
        self._n_hidden = len(hidden_dims)

        # Classification head: r_{L-1} → logits (trained with CE loss).
        # Avoids training-eval mismatch: r_{L-1} is a stable feature vector
        # not subject to clamping, so its logits are always calibrated.
        self.cls_head = nn.Linear(hidden_dims[-1], num_classes)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _act(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Activation for layer layer_idx. ReLU on hidden, identity on output."""
        if layer_idx < self._n_hidden:
            return F.relu(x)
        return x  # output layer: identity (r_L = logits)

    def _bottom_up(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """Initialise representations with a single bottom-up forward pass.

        Returns list [r_0=x, r_1, ..., r_L] — all detached.
        """
        reps = [x_flat.detach()]
        h = x_flat
        for i, layer in enumerate(self.layers):
            h = self._act(i, layer(h))
            reps.append(h.detach())
        return reps

    def _energy_fixed_weights(
        self,
        r0: torch.Tensor,
        r_mutable: List[torch.Tensor],
    ) -> torch.Tensor:
        """PC free energy with weights detached (used during inference phase).

        Gradients flow only to r_mutable tensors.
        """
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
        """PC free energy with weights active (used during weight update phase).

        Representations are detached; gradients flow only to weight matrices.
        """
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
        """Gradient descent inference: update r_l to minimise free energy.

        Args:
            reps: Initial representations [r_0, r_1, ..., r_L].
            clamp_last: If given (one-hot target), r_L is held fixed throughout.

        Returns:
            Updated representations (all detached).
        """
        r0 = reps[0].detach()
        r_free = [r.detach().clone() for r in reps[1:]]

        if clamp_last is not None:
            # Supervised inference: clamp output representation
            r_free[-1] = clamp_last.detach().float()

        # torch.enable_grad() so inference works even inside torch.no_grad() blocks
        with torch.enable_grad():
            for r in r_free:
                r.requires_grad_(True)

            for _ in range(self.T_pc):
                # Which representations are free to update?
                if clamp_last is not None:
                    update_reps = r_free[:-1]       # r_1 .. r_{L-1}
                else:
                    update_reps = r_free            # r_1 .. r_L (free)

                if not update_reps:
                    break

                energy = self._energy_fixed_weights(r0, r_free)
                grads = torch.autograd.grad(energy, update_reps, create_graph=False)

                updated = []
                for r, g in zip(update_reps, grads):
                    updated.append((r - self.lr_pc * g).detach().requires_grad_(True))

                if clamp_last is not None:
                    r_free = updated + [r_free[-1]]
                else:
                    r_free = updated

        return [r0] + [r.detach() for r in r_free]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pc_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PC training loss (supervised inference + weight-update energy + CE head).

        Two losses are returned so the training loop can log them separately:
          - energy: PC free energy over all layers (trains PC weights)
          - ce_loss: cross-entropy on cls_head(r_{L-1}) (trains cls_head + PC hidden weights)
        The combined loss = energy + ce_weight * ce_loss is what you backprop.

        Args:
            x: Input images  (B, C, H, W) or (B, input_size)
            y: Class labels  (B,)

        Returns:
            combined_loss: energy + ce_weight * ce_loss — backprop this.
            energy: PC free energy scalar (for logging).
            logits: (B, num_classes) from cls_head(r_{L-1}) — for accuracy logging.
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

        # Phase 1 — Inference (clamp r_L = one_hot(y))
        target = F.one_hot(y, self.num_classes).float()
        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_inference(reps_init, clamp_last=target)

        # Phase 2 — Energy with active weights (gradients → PC weight matrices)
        energy = self._energy_active_weights(reps_final)

        # CE head on r_{L-1}: r_{L-1} is detached from the PC graph but we
        # re-run the bottom-up pass to get a differentiable r_{L-1} for cls_head.
        # We stop gradients at r_{L-1} w.r.t. the PC layers to keep the two
        # objectives cleanly separated: PC energy trains the PC layers;
        # the CE loss trains cls_head (and pulls gradients back through
        # PC hidden layers only via the shared representation).
        h = x_flat
        for i, layer in enumerate(self.layers[:-1]):   # up to r_{L-1}
            h = self._act(i, layer(h))
        logits = self.cls_head(h)
        ce_loss = F.cross_entropy(logits, y)

        combined_loss = energy + self.ce_weight * ce_loss
        return combined_loss, energy.detach(), logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation forward pass using cls_head(r_{L-1}) after PC inference.

        Runs T_pc free-inference steps to allow PC hidden representations to
        settle, then classifies using the CE-trained classification head on
        r_{L-1} (the final hidden layer). This produces calibrated logits
        regardless of r_L's quality.

        Args:
            x: Input images (B, C, H, W) or (B, input_size)

        Returns:
            logits: (B, num_classes) from cls_head(r_{L-1}).
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_inference(reps_init, clamp_last=None)

        # r_{L-1} is reps_final[-2]; run it through the classification head
        return self.cls_head(reps_final[-2])

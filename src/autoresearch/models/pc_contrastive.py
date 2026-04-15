"""PC + Contrastive Learning model for MNIST.

Architecture (Iter 5 of anp_contrastive):
    PC Encoder (Rao & Ballard-style hierarchical predictive coding):
        Input (784) → [PC layers] → r_fc (256-dim representation)
        PC free inference: T_pc gradient-descent steps minimising free energy
        over internal representations {r_1, ..., r_{L-1}} (no clamping).
    Projection head (standard MLP):
        256 → FC(256) → BN → ReLU → FC(128)   (L2-normalised)
    Linear probe:
        256 → 10  (trained separately after pre-training)

Key differences from the failed SPC-FFNN (anp_spcnn iters 2-5):
  - No SNN components here — purely rate-coded PC encoder.
  - No shared weights between generative PC weights and any SNN pathway.
  - Contrastive objective (NT-Xent on projections) cleanly decouples the
    discriminative training from PC energy minimisation.
  - PC energy is an *optional* regulariser: if energy_weight=0, the model
    trains purely with the contrastive objective and the PC layers converge
    purely from the contrastive gradient flowing back through free inference.

Reference:
    Rao & Ballard (1999). Predictive coding in the visual cortex.
    Chen et al. (2020). A Simple Framework for Contrastive Learning.
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCContrastive(nn.Module):
    """PC-FFNN encoder with contrastive projection head.

    Args:
        input_size:   Flattened input dimension (784 for MNIST).
        hidden_dims:  PC encoder layer widths, e.g. [512, 256].
                      The last element is repr_dim.
        proj_hidden:  Projection head hidden dimension.
        proj_out:     Projection output dimension.
        num_classes:  Number of classes for the linear probe head.
        T_pc:         Number of PC free-inference gradient-descent steps.
        lr_pc:        Step size for representation updates during inference.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: List[int] = None,
        proj_hidden: int = 256,
        proj_out: int = 128,
        num_classes: int = 10,
        T_pc: int = 20,
        lr_pc: float = 0.05,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.input_size = input_size
        self.hidden_dims = hidden_dims
        self.repr_dim = hidden_dims[-1]
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self._n_layers = len(hidden_dims)

        # PC weight matrices W_l: r_{l-1} → r_l
        dims = [input_size] + hidden_dims
        self.pc_layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        # Projection head: non-linear MLP
        self.projection_head = nn.Sequential(
            nn.Linear(self.repr_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
        )

        # Linear probe head
        self.linear_probe = nn.Linear(self.repr_dim, num_classes)

    # ------------------------------------------------------------------
    # PC inference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _act(x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)

    def _bottom_up(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """Single bottom-up pass to initialise representations."""
        reps = [x_flat.detach()]
        h = x_flat
        for layer in self.pc_layers:
            h = self._act(layer(h)).detach()
            reps.append(h)
        return reps

    def _energy_fixed_weights(
        self,
        r0: torch.Tensor,
        r_mutable: List[torch.Tensor],
    ) -> torch.Tensor:
        """PC free energy with frozen weights (inference phase)."""
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable
        for i, layer in enumerate(self.pc_layers):
            w = layer.weight.detach()
            b = layer.bias.detach()
            mu = self._act(F.linear(all_r[i], w, b))
            energy = energy + 0.5 * ((all_r[i + 1] - mu) ** 2).sum()
        return energy

    def _run_free_inference(self, reps: List[torch.Tensor]) -> List[torch.Tensor]:
        """Free inference: T_pc steps of gradient descent on free energy.

        All representations are updated (no clamping).

        Args:
            reps: Initial representations [r_0, r_1, ..., r_{L-1}].

        Returns:
            Updated representations (all detached).
        """
        r0 = reps[0].detach()
        r_free = [r.detach().clone() for r in reps[1:]]

        with torch.enable_grad():
            for r in r_free:
                r.requires_grad_(True)

            for _ in range(self.T_pc):
                energy = self._energy_fixed_weights(r0, r_free)
                grads = torch.autograd.grad(energy, r_free, create_graph=False)
                r_free = [
                    (r - self.lr_pc * g).detach().requires_grad_(True)
                    for r, g in zip(r_free, grads)
                ]

        return [r0] + [r.detach() for r in r_free]

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """PC encode images to representations via free inference.

        Runs T_pc free-inference steps and returns the final
        r_{L-1} representation (repr_dim-dimensional).

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            (N, repr_dim) representations.
        """
        x_flat = x.view(x.size(0), -1)
        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_free_inference(reps_init)
        return reps_final[-1]  # r_{L-1}: (N, repr_dim)

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Project to the contrastive embedding space (L2-normalised).

        Args:
            h: (N, repr_dim) representations.

        Returns:
            (N, proj_out) L2-normalised projections.
        """
        return F.normalize(self.projection_head(h), dim=-1)

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        """Linear probe classification.

        Args:
            h: (N, repr_dim) representations.

        Returns:
            (N, num_classes) logits.
        """
        return self.linear_probe(h)

    def pc_energy(self, x: torch.Tensor) -> torch.Tensor:
        """Compute PC free energy on current representations (for regularisation).

        Performs a bottom-up pass (differentiable) and computes the energy
        with active weights.  Use as an optional regulariser during
        contrastive pre-training: loss = NT-Xent + energy_weight * pc_energy.

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            Scalar free energy.
        """
        B = x.size(0)
        x_flat = x.view(B, -1)
        energy = x_flat.new_zeros(())
        h = x_flat
        reps = [h]
        for layer in self.pc_layers:
            h = self._act(layer(h))
            reps.append(h)
        for i, layer in enumerate(self.pc_layers):
            mu = self._act(layer(reps[i].detach()))
            e = reps[i + 1].detach() - mu
            energy = energy + 0.5 * (e ** 2).sum()
        return energy / B  # per-sample energy

    # ------------------------------------------------------------------
    # Standard forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then classify.

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            (N, num_classes) logits.
        """
        return self.classify(self.encode(x))

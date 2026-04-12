"""Predictive Coding Convolutional Neural Network (PC-CNN).

Extends hierarchical predictive coding (Rao & Ballard, 1999) to convolutional
feature extraction. Each conv block produces a spatial representation r_l; the
PC inference loop updates these representations to minimise the free energy:

  E = 0.5*||r_1 - pool(ReLU(conv_0(r_0)))||²
    + 0.5*||r_2 - pool(ReLU(conv_1(r_1)))||²
    + ...
    + 0.5*||r_fc - ReLU(fc(flatten(r_{n_conv})))||²
    + 0.5*||r_out - pc_out(r_fc)||²

Representations live in the spatial domain (B, C, H, W) for conv layers and
in the vector domain (B, D) for the FC hidden and output layers.

Training (identical two-phase pattern to PC-FFNN):
  Phase 1 — Inference: minimise E over {r_1, ..., r_fc} (weights fixed, T_pc steps).
             Supervised: clamp r_out = one_hot(y).
  Phase 2 — Weight update: backprop through E and CE on cls_head(r_fc).

Evaluation:
  cls_head(r_fc) after T_pc free-inference steps (no clamping).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCCNN(nn.Module):
    """Predictive Coding CNN for image classification.

    Args:
        in_channels:   Input image channels (1 for MNIST grayscale).
        input_hw:      Input spatial size in pixels (28 for MNIST 28×28).
        conv_channels: Output channels per conv block, e.g. [32, 64].
        kernel_size:   Convolution kernel size (same for all blocks).
        fc_hidden:     Width of the FC hidden representation (r_fc).
        num_classes:   Number of output classes.
        T_pc:          Inference steps per forward pass.
        lr_pc:         Step size for representation updates during inference.
        ce_weight:     Weight of CE loss relative to PC energy loss.
    """

    def __init__(
        self,
        in_channels: int = 1,
        input_hw: int = 28,
        conv_channels: Optional[List[int]] = None,
        kernel_size: int = 5,
        fc_hidden: int = 256,
        num_classes: int = 10,
        T_pc: int = 20,
        lr_pc: float = 0.05,
        ce_weight: float = 1.0,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64]

        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight
        self.num_classes = num_classes
        self._n_conv = len(conv_channels)

        # Conv blocks: each produces a spatial representation r_l
        channels = [in_channels] + list(conv_channels)
        self.convs = nn.ModuleList(
            [nn.Conv2d(channels[i], channels[i + 1], kernel_size) for i in range(len(channels) - 1)]
        )
        self.pool = nn.MaxPool2d(2, 2)

        # Compute flattened size dynamically for the given input_hw
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_hw, input_hw)
            for conv in self.convs:
                dummy = self.pool(F.relu(conv(dummy)))
            self.flattened_size = int(dummy.view(1, -1).shape[1])

        # FC layer: flatten(r_{n_conv}) → r_fc
        self.fc = nn.Linear(self.flattened_size, fc_hidden)

        # PC output layer: r_fc → r_out  (part of the PC energy graph)
        self.pc_out = nn.Linear(fc_hidden, num_classes)

        # Classification head: r_fc → logits  (auxiliary CE head, same as cls_head in PCFFNN)
        self.cls_head = nn.Linear(fc_hidden, num_classes)

    # --------------------------------------------------------------------------
    # Low-level helpers
    # --------------------------------------------------------------------------

    def _conv_block(self, r: torch.Tensor, i: int, detach_weights: bool) -> torch.Tensor:
        """Apply conv block i: pool(ReLU(conv_i(r)))."""
        conv = self.convs[i]
        if detach_weights:
            h = F.conv2d(r, conv.weight.detach(), conv.bias.detach(),
                         stride=conv.stride, padding=conv.padding)
        else:
            h = conv(r)
        return self.pool(F.relu(h))

    def _fc_block(self, r_flat: torch.Tensor, detach_weights: bool) -> torch.Tensor:
        """Apply FC: ReLU(fc(r_flat))."""
        if detach_weights:
            return F.relu(F.linear(r_flat, self.fc.weight.detach(), self.fc.bias.detach()))
        return F.relu(self.fc(r_flat))

    def _pc_out_block(self, r_fc: torch.Tensor, detach_weights: bool) -> torch.Tensor:
        """Apply pc_out: pc_out(r_fc)  — identity activation (output layer)."""
        if detach_weights:
            return F.linear(r_fc, self.pc_out.weight.detach(), self.pc_out.bias.detach())
        return self.pc_out(r_fc)

    # --------------------------------------------------------------------------
    # PC core: bottom-up init, energy, inference
    # --------------------------------------------------------------------------

    def _bottom_up(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Single feedforward pass to initialise representations.

        Returns [r_0, r_1, ..., r_{n_conv}, r_fc, r_out] — all detached.
        """
        reps = [x.detach()]
        h = x
        for i in range(self._n_conv):
            h = self._conv_block(h, i, detach_weights=False)
            reps.append(h.detach())
        h_flat = h.view(h.shape[0], -1)
        r_fc = self._fc_block(h_flat, detach_weights=False)
        reps.append(r_fc.detach())
        r_out = self._pc_out_block(r_fc, detach_weights=False)
        reps.append(r_out.detach())
        return reps

    def _energy_fixed_weights(
        self,
        r0: torch.Tensor,
        r_mutable: List[torch.Tensor],
    ) -> torch.Tensor:
        """PC free energy with weights detached — gradients flow to r_mutable only."""
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable

        # Conv layers: E_i = 0.5*||r_{i+1} - pool(ReLU(conv_i(r_i)))||²
        for i in range(self._n_conv):
            mu = self._conv_block(all_r[i], i, detach_weights=True)
            e = all_r[i + 1] - mu
            energy = energy + 0.5 * (e ** 2).sum()

        # FC layer: E_fc = 0.5*||r_fc - ReLU(fc(flatten(r_{n_conv})))||²
        r_conv_last = all_r[self._n_conv]
        r_flat = r_conv_last.view(r_conv_last.shape[0], -1)
        mu_fc = self._fc_block(r_flat, detach_weights=True)
        e_fc = all_r[self._n_conv + 1] - mu_fc
        energy = energy + 0.5 * (e_fc ** 2).sum()

        # PC output layer: E_out = 0.5*||r_out - pc_out(r_fc)||²
        mu_out = self._pc_out_block(all_r[self._n_conv + 1], detach_weights=True)
        e_out = all_r[self._n_conv + 2] - mu_out
        energy = energy + 0.5 * (e_out ** 2).sum()

        return energy

    def _energy_active_weights(self, reps: List[torch.Tensor]) -> torch.Tensor:
        """PC free energy with weights active — gradients flow to weights only."""
        energy = reps[0].new_zeros(())

        # Conv layers
        for i in range(self._n_conv):
            mu = self._conv_block(reps[i].detach(), i, detach_weights=False)
            e = reps[i + 1].detach() - mu
            energy = energy + 0.5 * (e ** 2).sum()

        # FC layer
        r_conv_last = reps[self._n_conv].detach()
        r_flat = r_conv_last.view(r_conv_last.shape[0], -1)
        mu_fc = self._fc_block(r_flat, detach_weights=False)
        e_fc = reps[self._n_conv + 1].detach() - mu_fc
        energy = energy + 0.5 * (e_fc ** 2).sum()

        # PC output layer
        mu_out = self._pc_out_block(reps[self._n_conv + 1].detach(), detach_weights=False)
        e_out = reps[self._n_conv + 2].detach() - mu_out
        energy = energy + 0.5 * (e_out ** 2).sum()

        return energy

    def _run_inference(
        self,
        reps: List[torch.Tensor],
        clamp_last: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Gradient descent inference: update {r_1, ..., r_fc} to minimise free energy.

        Args:
            reps:       Initial representations [r_0, r_1, ..., r_fc, r_out].
            clamp_last: If given (one-hot target), r_out is held fixed throughout.

        Returns:
            Updated representations (all detached).
        """
        r0 = reps[0].detach()
        r_free = [r.detach().clone() for r in reps[1:]]

        if clamp_last is not None:
            r_free[-1] = clamp_last.detach().float()

        with torch.enable_grad():
            for r in r_free:
                r.requires_grad_(True)

            for _ in range(self.T_pc):
                if clamp_last is not None:
                    update_reps = r_free[:-1]   # r_1 .. r_fc  (r_out clamped)
                else:
                    update_reps = r_free         # r_1 .. r_out (all free)

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

    # --------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------

    def pc_loss(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        energy_weight: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PC training loss (supervised inference + energy + CE head).

        Args:
            x: Input images  (B, C, H, W)
            y: Class labels  (B,)

        Returns:
            combined_loss:
                - legacy mode (energy_weight is None):
                    energy + ce_weight * ce_loss
                - scheduled mode (energy_weight provided):
                    energy_weight * (energy / B) + (1 - energy_weight) * ce_loss
            energy:        PC free energy scalar (detached, for logging).
            logits:        (B, num_classes) from cls_head(r_fc) (detached, for logging).
        """
        B = x.shape[0]
        target = F.one_hot(y, self.num_classes).float()
        reps_init = self._bottom_up(x)
        reps_final = self._run_inference(reps_init, clamp_last=target)

        energy = self._energy_active_weights(reps_final)

        # CE head: re-run conv stack differentiably to get r_fc with gradients to weights.
        h = x
        for i in range(self._n_conv):
            h = self._conv_block(h, i, detach_weights=False)
        h_flat = h.view(h.shape[0], -1)
        r_fc = self._fc_block(h_flat, detach_weights=False)
        logits = self.cls_head(r_fc)
        ce_loss = F.cross_entropy(logits, y)

        if energy_weight is None:
            # Backward-compatible path used by iter11.
            combined_loss = energy + self.ce_weight * ce_loss
        else:
            # Scheduled convex blend on per-sample energy scale.
            energy_norm = energy / B
            combined_loss = energy_weight * energy_norm + (1.0 - energy_weight) * ce_loss
        return combined_loss, energy.detach(), logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation forward pass: cls_head(r_fc) after PC free inference.

        Args:
            x: Input images  (B, C, H, W)

        Returns:
            logits: (B, num_classes)
        """
        reps_init = self._bottom_up(x)
        reps_final = self._run_inference(reps_init, clamp_last=None)
        # r_fc is reps_final[-2] (last hidden rep before r_out)
        return self.cls_head(reps_final[-2])

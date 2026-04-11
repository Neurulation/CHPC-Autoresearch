"""Predictive Coding Encoder-Decoder Network (PC-EncDec).

Extends PC-FFNN with bidirectional PC: explicit top-down decoder weights
complement the bottom-up encoder, creating a full generative model.

Architecture:
    Encoder (bottom-up): r_0(=x) → r_1 → ... → r_L (classification head)
    Decoder (top-down): r_L → ... → r_1 → r_0'(=x_hat, reconstruction)

    Only hidden layers participate in decoding — the classification output
    r_L (clamped to one_hot(y) during training) is not decoded back, keeping
    the classification objective cleanly separate from the generative path.

PC Free Energy:
    F = Σ_l 0.5 * ||r_l - f(W_enc[l-1] @ r_{l-1})||²        (encoder errors)
      + dec_weight * Σ_j 0.5 * ||r_j - g(W_dec[j] @ r_{j+1})||²  (decoder errors)

    where:
      - W_enc[l]: enc_layers[l], maps enc_dims[l] → enc_dims[l+1]
      - W_dec[j]: dec_layers[j], maps dec_dims[j+1] → dec_dims[j]
      - enc_dims = [input_size] + hidden_dims + [num_classes]
      - dec_dims = [input_size] + hidden_dims  (no classification layer in decoder)

Skip connections as shared representation:
    Each representation r_l is updated during inference to minimise *both*
    the encoder error (r_l predicted from r_{l-1}) and the decoder error
    (r_{l-1} predicted from r_l). Shared r_l mediates the skip connection:
    information from the encoder path directly shapes the decoder prediction
    target, and vice versa — the bidirectional constraint drives r_l to be
    simultaneously recognisable from below and generative toward below.

Training (matches train_pcnn.py interface):
    Phase 1 — Inference: T_pc gradient steps on r_1..r_{L-1} (r_0=x clamped,
              r_L=one_hot(y) clamped during training) minimising combined
              enc+dec free energy (weights fixed).
    Phase 2 — Weight update: PC energy (enc_layers + dec_layers active, reps
              detached) + CE loss via forward pass through enc_layers+cls_head.

References:
    - Rao & Ballard (1999). Predictive coding in the visual cortex.
    - Hinton & Zemel (1994). Autoencoders, minimum description length.
    - Friston (2005). A theory of cortical responses (bidirectional PC).
    - Ronneberger et al. (2015). U-Net (skip connections in enc-dec).
"""

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCEncDec(nn.Module):
    """Predictive Coding Encoder-Decoder for image classification.

    Args:
        input_size: Flattened input dimension (784 for MNIST)
        hidden_dims: Hidden layer widths, e.g. [256, 128]
        num_classes: Number of output classes
        T_pc: Inference steps per forward pass
        lr_pc: Step size for representation updates during inference
        ce_weight: Cross-entropy loss weight relative to PC energy
        dec_weight: Decoder energy weight relative to encoder energy
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 10,
        T_pc: int = 20,
        lr_pc: float = 0.05,
        ce_weight: float = 1.0,
        dec_weight: float = 0.5,
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_dims = list(hidden_dims)
        self.num_classes = num_classes
        self.T_pc = T_pc
        self.lr_pc = lr_pc
        self.ce_weight = ce_weight
        self.dec_weight = dec_weight

        self._n_hidden = len(hidden_dims)

        # Encoder: [input_size, hidden_0, ..., hidden_L-1, num_classes]
        enc_dims = [input_size] + self.hidden_dims + [num_classes]
        self.enc_layers = nn.ModuleList(
            [nn.Linear(enc_dims[i], enc_dims[i + 1]) for i in range(len(enc_dims) - 1)]
        )

        # Decoder: [input_size, hidden_0, ..., hidden_L-1]
        # dec_layers[j] maps dec_dims[j+1] → dec_dims[j]:
        #   dec_layers[0]: hidden_0 → input_size  (reconstruction)
        #   dec_layers[1]: hidden_1 → hidden_0
        #   ...
        dec_dims = [input_size] + self.hidden_dims
        self.dec_layers = nn.ModuleList(
            [nn.Linear(dec_dims[j + 1], dec_dims[j]) for j in range(len(dec_dims) - 1)]
        )

        # Classification head: trained with CE loss on the final hidden rep.
        self.cls_head = nn.Linear(self.hidden_dims[-1], num_classes)

    # ------------------------------------------------------------------
    # Activation helpers
    # ------------------------------------------------------------------

    def _enc_act(self, layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Encoder activation: ReLU for hidden layers, identity for output."""
        if layer_idx < self._n_hidden:
            return F.relu(x)
        return x

    def _dec_act(self, dec_layer_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Decoder activation: identity for input reconstruction (idx=0), ReLU otherwise."""
        if dec_layer_idx == 0:
            return x  # input reconstruction: linear decoder
        return F.relu(x)

    # ------------------------------------------------------------------
    # Initialisation: single bottom-up pass
    # ------------------------------------------------------------------

    def _bottom_up(self, x_flat: torch.Tensor) -> List[torch.Tensor]:
        """Initialise representations with a single bottom-up forward pass.

        Returns list [r_0=x, r_1, ..., r_L] — all detached.
        """
        reps = [x_flat.detach()]
        h = x_flat
        for i, layer in enumerate(self.enc_layers):
            h = self._enc_act(i, layer(h))
            reps.append(h.detach())
        return reps

    # ------------------------------------------------------------------
    # Energy: combined encoder + decoder
    # ------------------------------------------------------------------

    def _energy_fixed_weights(
        self,
        r0: torch.Tensor,
        r_mutable: List[torch.Tensor],
    ) -> torch.Tensor:
        """Combined enc+dec free energy with weights detached (inference phase).

        Gradients flow only to r_mutable tensors (r_1 .. r_L).
        """
        energy = r0.new_zeros(())
        all_r = [r0] + r_mutable            # all_r[l] = r_l

        # Encoder errors: ||r_{l+1} - f(enc[l] @ r_l)||²
        for i, layer in enumerate(self.enc_layers):
            w, b = layer.weight.detach(), layer.bias.detach()
            mu_enc = self._enc_act(i, F.linear(all_r[i], w, b))
            e_enc = all_r[i + 1] - mu_enc
            energy = energy + 0.5 * (e_enc ** 2).sum()

        # Decoder errors: ||r_j - g(dec[j] @ r_{j+1})||²  for j=0..n_hidden-1
        for j, layer in enumerate(self.dec_layers):
            w, b = layer.weight.detach(), layer.bias.detach()
            mu_dec = self._dec_act(j, F.linear(all_r[j + 1], w, b))
            e_dec = all_r[j] - mu_dec
            energy = energy + self.dec_weight * 0.5 * (e_dec ** 2).sum()

        return energy

    def _energy_active_weights(self, reps: List[torch.Tensor]) -> torch.Tensor:
        """Combined enc+dec energy with active weights (weight update phase).

        Representations detached; gradients flow to enc_layers and dec_layers.
        """
        energy = reps[0].new_zeros(())

        # Encoder errors
        for i, layer in enumerate(self.enc_layers):
            mu_enc = self._enc_act(i, layer(reps[i].detach()))
            e_enc = reps[i + 1].detach() - mu_enc
            energy = energy + 0.5 * (e_enc ** 2).sum()

        # Decoder errors
        for j, layer in enumerate(self.dec_layers):
            mu_dec = self._dec_act(j, layer(reps[j + 1].detach()))
            e_dec = reps[j].detach() - mu_dec
            energy = energy + self.dec_weight * 0.5 * (e_dec ** 2).sum()

        return energy

    # ------------------------------------------------------------------
    # Inference: T_pc gradient steps on representations
    # ------------------------------------------------------------------

    def _run_inference(
        self,
        reps: List[torch.Tensor],
        clamp_last: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Gradient descent inference: update representations to minimise enc+dec energy.

        Args:
            reps: Initial representations [r_0, r_1, ..., r_L] (all detached).
            clamp_last: If given (one-hot target), r_L is held fixed.

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
                update_reps = r_free[:-1] if clamp_last is not None else r_free

                if not update_reps:
                    break

                energy = self._energy_fixed_weights(r0, r_free)
                grads = torch.autograd.grad(energy, update_reps, create_graph=False)

                updated = [
                    (r - self.lr_pc * g).detach().requires_grad_(True)
                    for r, g in zip(update_reps, grads)
                ]

                if clamp_last is not None:
                    r_free = updated + [r_free[-1]]
                else:
                    r_free = updated

        return [r0] + [r.detach() for r in r_free]

    # ------------------------------------------------------------------
    # Public API (matches train_pcnn.py interface)
    # ------------------------------------------------------------------

    def pc_loss(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute PC-EncDec training loss.

        Phase 1 — Inference: settle r_1..r_{L-1} (r_L clamped, weights fixed).
        Phase 2 — Weight update: enc+dec energy (active weights, reps detached)
                                 + CE via bottom-up forward through enc_layers + cls_head.

        Returns:
            combined_loss: energy + ce_weight * ce_loss
            energy: PC free energy scalar (logging)
            logits: (B, num_classes) from cls_head(r_{L-1})
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

        target = F.one_hot(y, self.num_classes).float()
        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_inference(reps_init, clamp_last=target)

        energy = self._energy_active_weights(reps_final)

        # CE head: differentiable forward through enc_layers[:-1] → cls_head
        h = x_flat
        for i, layer in enumerate(self.enc_layers[:-1]):   # up to r_{L-1}
            h = self._enc_act(i, layer(h))
        logits = self.cls_head(h)
        ce_loss = F.cross_entropy(logits, y)

        combined_loss = energy + self.ce_weight * ce_loss
        return combined_loss, energy.detach(), logits.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluation: free PC inference, classify via cls_head(r_{L-1}).

        Args:
            x: Input images (B, C, H, W) or (B, input_size)

        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]
        x_flat = x.view(B, -1)

        reps_init = self._bottom_up(x_flat)
        reps_final = self._run_inference(reps_init, clamp_last=None)

        # reps_final[-2] = r_{L-1} (the last hidden layer, index n_hidden)
        return self.cls_head(reps_final[-2])

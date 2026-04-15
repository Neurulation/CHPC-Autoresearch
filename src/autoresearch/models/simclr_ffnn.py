"""SimCLR with MLP encoder backbone for contrastive learning on MNIST.

Architecture (Iter 1 of anp_contrastive):
    Encoder:          784 → FC(512) → BN → ReLU → FC(256) → BN → ReLU
                      Output: repr_dim=256 representations.
    Projection head:  256 → FC(256) → BN → ReLU → FC(128)  (L2-normalised)
    Linear probe:     256 → 10  (trained separately after pre-training)

Usage:
    Pre-training:  h = model.encode(x); z = model.project(h)
                   NT-Xent loss on z pairs.
    Linear probe:  h = model.encode(x).detach(); logits = model.classify(h)
    Evaluation:    model(x)  →  logits  (encode → classify)

Reference:
    Chen et al. (2020). A Simple Framework for Contrastive Learning of
    Visual Representations.  https://arxiv.org/abs/2002.05709
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRFFNN(nn.Module):
    """SimCLR with MLP encoder for contrastive pre-training on MNIST.

    Args:
        input_size:   Flattened input dimension (784 for MNIST 28×28).
        encoder_dims: Hidden layer widths for the encoder MLP, e.g. [512, 256].
                      The last element is repr_dim.
        proj_hidden:  Projection head hidden dimension.
        proj_out:     Projection head output dimension (contrastive space).
        num_classes:  Number of classes for the linear probe head.
        dropout:      Dropout probability in the encoder (0 = disabled).
    """

    def __init__(
        self,
        input_size: int = 784,
        encoder_dims: List[int] = None,
        proj_hidden: int = 256,
        proj_out: int = 128,
        num_classes: int = 10,
        dropout: float = 0.0,
    ):
        super().__init__()

        if encoder_dims is None:
            encoder_dims = [512, 256]

        self.input_size = input_size
        self.repr_dim = encoder_dims[-1]

        # Encoder: MLP backbone with BatchNorm
        enc_layers = []
        in_dim = input_size
        for out_dim in encoder_dims:
            enc_layers.append(nn.Linear(in_dim, out_dim))
            enc_layers.append(nn.BatchNorm1d(out_dim))
            enc_layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            in_dim = out_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Projection head: non-linear MLP
        self.projection_head = nn.Sequential(
            nn.Linear(self.repr_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
        )

        # Linear probe head (frozen encoder + this layer during probing)
        self.linear_probe = nn.Linear(self.repr_dim, num_classes)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images to representations.

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            (N, repr_dim) representations.
        """
        return self.encoder(x.view(x.size(0), -1))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Project representations to the contrastive embedding space.

        Args:
            h: (N, repr_dim) representations.

        Returns:
            (N, proj_out) L2-normalised projections.
        """
        return F.normalize(self.projection_head(h), dim=-1)

    def classify(self, h: torch.Tensor) -> torch.Tensor:
        """Classify via the linear probe head.

        Args:
            h: (N, repr_dim) representations.

        Returns:
            (N, num_classes) logits.
        """
        return self.linear_probe(h)

    # ------------------------------------------------------------------
    # Standard forward (used by the generic validate() loop)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode then classify.

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            (N, num_classes) logits.
        """
        return self.classify(self.encode(x))

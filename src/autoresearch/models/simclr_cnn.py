"""SimCLR with CNN encoder backbone for contrastive learning on MNIST.

Architecture (Iter 2 of anp_contrastive):
    CNN Encoder:
        Conv1: 1→32, 3×3 pad=1 → BN → ReLU → MaxPool(2×2) → (N,32,14,14)
        Conv2: 32→64, 3×3 pad=1 → BN → ReLU → MaxPool(2×2) → (N,64,7,7)
        Flatten → FC(3136→256) → BN → ReLU     repr_dim = 256
    Projection head:  256 → FC(256) → BN → ReLU → FC(128)  (L2-normalised)
    Linear probe:     256 → 10  (trained separately after pre-training)

Usage identical to SimCLRFFNN — encode / project / classify / forward.

Reference:
    Chen et al. (2020). A Simple Framework for Contrastive Learning of
    Visual Representations.  https://arxiv.org/abs/2002.05709
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRCNN(nn.Module):
    """SimCLR with CNN encoder for contrastive pre-training on MNIST.

    Args:
        input_channels: Input image channels (1 for grayscale MNIST).
        repr_dim:        Encoder output dimensionality.
        proj_hidden:     Projection head hidden dimension.
        proj_out:        Projection head output dimension (contrastive space).
        num_classes:     Number of classes for the linear probe head.
    """

    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 256,
        proj_hidden: int = 256,
        proj_out: int = 128,
        num_classes: int = 10,
    ):
        super().__init__()

        self.repr_dim = repr_dim

        # CNN backbone with BatchNorm
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        # After two 2×2 MaxPools on 28×28: spatial size = 7×7
        self.fc_enc = nn.Linear(64 * 7 * 7, repr_dim)
        self.bn_fc = nn.BatchNorm1d(repr_dim)

        # Projection head: non-linear MLP
        self.projection_head = nn.Sequential(
            nn.Linear(repr_dim, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
        )

        # Linear probe head
        self.linear_probe = nn.Linear(repr_dim, num_classes)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """CNN encode images to representations.

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            (N, repr_dim) representations.
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return F.relu(self.bn_fc(self.fc_enc(x)))

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """Project representations to the contrastive embedding space.

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

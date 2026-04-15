"""SNN SimCLR: Spiking CNN encoder + MLP projection head.

Architecture (Iter 4 of anp_contrastive):
    SNN CNN Encoder (matching topology of SNNCNN from anp_snn iter 1):
        Rate-encode input → (T, N, 1, 28, 28) binary spikes
        Conv1(1→32,3×3) → LIF → MaxPool(2×2) → (T, N, 32, 14, 14) spikes
        Conv2(32→64,3×3) → LIF → MaxPool(2×2) → (T, N, 64, 7, 7) spikes
        Flatten → FC(3136→128) → LIF → (T, N, 128) spikes
        Sum over T → (N, 128) float spike-accumulation representation
    Projection head (standard MLP):
        128 → FC(256) → BN → ReLU → FC(128)   (L2-normalised)
    Linear probe:
        128 → 10  (trained separately after pre-training)

As with SNNSimCLRFFNN, augmentations are applied to float images before
spike encoding so that each view receives independent spike realisations.

Reference:
    Chen et al. (2020). A Simple Framework for Contrastive Learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.spike_encoding import rate_encode, ttfs_encode


class SNNSimCLRCNN(nn.Module):
    """SimCLR with SNN-CNN encoder for contrastive pre-training on MNIST.

    Args:
        input_channels: Input image channels (1 for grayscale MNIST).
        repr_dim:        Size of the FC hidden layer / representation.
        proj_hidden:     Projection head hidden dimension.
        proj_out:        Projection head output dimension.
        num_classes:     Number of classes for the linear probe head.
        beta:            LIF membrane decay constant (0 < β < 1).
        threshold:       LIF firing threshold.
        timesteps:       Number of SNN simulation timesteps T.
        encoding:        Spike encoding: 'rate' or 'ttfs'.
    """

    def __init__(
        self,
        input_channels: int = 1,
        repr_dim: int = 128,
        proj_hidden: int = 256,
        proj_out: int = 128,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 0.9,
        timesteps: int = 25,
        encoding: str = "rate",
    ):
        super().__init__()

        self.timesteps = timesteps
        self.encoding = encoding
        self.repr_dim = repr_dim

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Conv blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # After two 2×2 MaxPools on 28×28: 64 × 7 × 7 = 3136
        self.fc1 = nn.Linear(64 * 7 * 7, repr_dim)

        # LIF neurons — one per layer
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Projection head
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
        """SNN-CNN encode images to spike-accumulation representations.

        Args:
            x: Image batch (N, 1, H, W).

        Returns:
            (N, repr_dim) float spike-accumulation representations.
        """
        if self.encoding == "ttfs":
            spikes = ttfs_encode(x, self.timesteps)
        else:
            spikes = rate_encode(x, self.timesteps)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        spk3_acc = torch.zeros(x.size(0), self.repr_dim, device=x.device)

        for t in range(self.timesteps):
            cur1 = self.conv1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1 = self.pool1(spk1)

            cur2 = self.conv2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2 = self.pool2(spk2)

            cur3 = self.fc1(spk2.view(spk2.size(0), -1))
            spk3, mem3 = self.lif3(cur3, mem3)
            spk3_acc = spk3_acc + spk3

        return spk3_acc  # (N, repr_dim) float

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

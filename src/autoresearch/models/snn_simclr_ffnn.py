"""SNN SimCLR: Spiking FFNN encoder + MLP projection head.

Architecture (Iter 3 of anp_contrastive):
    SNN Encoder (fully spiking FFNN, snntorch Leaky LIF):
        Rate-encode input → (T, N, 784) binary spikes
        FC(784→512) → LIF → binary spikes (T, N, 512)
        FC(512→256) → LIF → binary spikes (T, N, 256)
        Sum over T → (N, 256) float spike-accumulation representations
    Projection head (standard MLP):
        256 → FC(256) → BN → ReLU → FC(128)   (L2-normalised)
    Linear probe:
        256 → 10  (trained separately after pre-training)

Key insight: augmentations are applied to float images *before* spike encoding
so that the two views see different spike realisations.  The contrastive loss
operates on the float-valued spike accumulations — not on raw binary spikes —
which makes gradient flow through the surrogate well-defined.

Reference:
    Chen et al. (2020). A Simple Framework for Contrastive Learning.
    Maass (1997). Networks of spiking neurons: the third generation of
                  neural network models.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.spike_encoding import rate_encode, ttfs_encode


class SNNSimCLRFFNN(nn.Module):
    """SimCLR with SNN-FFNN encoder for contrastive pre-training on MNIST.

    Args:
        input_size:  Flattened input dimension (784 for MNIST 28×28).
        hidden1:     First hidden SNN layer size.
        hidden2:     Second hidden SNN layer size; also the repr_dim.
        proj_hidden: Projection head hidden dimension.
        proj_out:    Projection head output dimension.
        num_classes: Number of classes for the linear probe head.
        beta:        LIF membrane decay constant (0 < β < 1).
        threshold:   LIF firing threshold.
        timesteps:   Number of SNN simulation timesteps T.
        encoding:    Spike encoding scheme: 'rate' (Bernoulli) or 'ttfs'.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 512,
        hidden2: int = 256,
        proj_hidden: int = 256,
        proj_out: int = 128,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 0.9,
        timesteps: int = 25,
        encoding: str = "rate",
    ):
        super().__init__()

        self.input_size = input_size
        self.timesteps = timesteps
        self.encoding = encoding
        self.repr_dim = hidden2

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # SNN encoder layers
        self.fc1 = nn.Linear(input_size, hidden1)
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        # Projection head (standard MLP, operates on float spike accumulations)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden2, proj_hidden),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, proj_out),
        )

        # Linear probe head
        self.linear_probe = nn.Linear(hidden2, num_classes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _spike_encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode flat image tensor to spike trains.

        Args:
            x: (N, input_size) normalised pixel values in [0, 1].

        Returns:
            (T, N, input_size) binary spike tensor.
        """
        if self.encoding == "ttfs":
            return ttfs_encode(x, self.timesteps)
        return rate_encode(x, self.timesteps)

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """SNN encode images to spike-accumulation representations.

        Runs T timesteps of the spiking FFNN and returns the sum of
        output spikes (a float tensor suitable for contrastive training).

        Args:
            x: Image batch (N, C, H, W).

        Returns:
            (N, repr_dim) float spike-accumulation representations.
        """
        x_flat = x.view(x.size(0), -1)
        spikes = self._spike_encode(x_flat)

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk2_acc = torch.zeros(x.size(0), self.repr_dim, device=x.device)

        for t in range(self.timesteps):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_acc = spk2_acc + spk2

        return spk2_acc  # (N, repr_dim) float

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

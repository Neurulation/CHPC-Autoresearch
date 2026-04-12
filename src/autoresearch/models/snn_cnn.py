"""Spiking Convolutional Neural Network — Phase B of the ANP SNN project.

Spiking analogue of the rate-coded CNN baseline (anp: image_processing_nn iter 1).
Uses LIF neurons in place of ReLU activations. Same spatial topology: two
Conv+Pool blocks followed by two FC layers.

Architecture (matching rate-coded CNN)
---------------------------------------
Input  : (N, 1, 28, 28) MNIST images, rate-encoded over T timesteps
Conv1  : Conv2d(1→32, 3x3, pad=1) → LIF → MaxPool(2x2)  → (N, 32, 14, 14)
Conv2  : Conv2d(32→64, 3x3, pad=1) → LIF → MaxPool(2x2) → (N, 64, 7, 7)
Flatten:                                                  → (N, 3136)
FC1    : Linear(3136→128) → LIF                          → (N, 128)
FC2    : Linear(128→10) → LIF                            → (N, 10)

Rate coding: pixel intensities → Bernoulli spike probabilities over T steps.
Output: spike-count logits (sum over T), compatible with nn.CrossEntropyLoss.

Total parameters: ~421,642 (vs 421,642 for the rate-coded CNN — identical).
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.spike_encoding import rate_encode, ttfs_encode


class SNNCNN(nn.Module):
    """Spiking CNN for MNIST — direct spiking analogue of the rate-coded CNN.

    Args:
        input_channels: Number of input image channels (1 for grayscale MNIST).
        num_classes:    Number of output classes.
        beta:           LIF membrane decay constant (0 < beta < 1).
        threshold:      LIF firing threshold.
        timesteps:      Number of simulation timesteps T for rate coding.
        encoding:       Spike encoding scheme — 'rate' (Bernoulli) or 'ttfs'
                        (time-to-first-spike). Default: 'rate'.
    """

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        timesteps: int = 25,
        encoding: str = "rate",
    ) -> None:
        super().__init__()

        self.timesteps = timesteps
        self.encoding = encoding
        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Conv blocks
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # FC layers  (64 * 7 * 7 = 3136 after two 2x2 pools on 28x28)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

        # LIF neurons — one per layer
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif4 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image batch to spike trains using the configured scheme.

        Args:
            x: (N, C, H, W) image batch with pixel values in [0, 1].

        Returns:
            (T, N, C, H, W) binary spike tensor.
        """
        if self.encoding == "ttfs":
            return ttfs_encode(x, self.timesteps)
        return rate_encode(x, self.timesteps)

    def _forward_snn(
        self, spikes: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the SNN forward pass over T timesteps.

        Args:
            spikes: (T, N, C, H, W) rate-coded input.

        Returns:
            - spk4_rec: (T, N, num_classes) output spike trains
            - recordings: dict of per-layer spikes and membrane potentials
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        mem4 = self.lif4.init_leaky()

        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        spk3_rec, mem3_rec = [], []
        spk4_rec, mem4_rec = [], []

        for t in range(self.timesteps):
            # Conv block 1
            cur1 = self.conv1(spikes[t])          # (N, 32, 28, 28)
            spk1, mem1 = self.lif1(cur1, mem1)    # (N, 32, 28, 28)
            spk1 = self.pool1(spk1)               # (N, 32, 14, 14)

            # Conv block 2
            cur2 = self.conv2(spk1)               # (N, 64, 14, 14)
            spk2, mem2 = self.lif2(cur2, mem2)    # (N, 64, 14, 14)
            spk2 = self.pool2(spk2)               # (N, 64,  7,  7)

            # FC block
            flat = spk2.view(spk2.size(0), -1)    # (N, 3136)
            cur3 = self.fc1(flat)                 # (N, 128)
            spk3, mem3 = self.lif3(cur3, mem3)

            cur4 = self.fc2(spk3)                 # (N, 10)
            spk4, mem4 = self.lif4(cur4, mem4)

            spk1_rec.append(spk1); mem1_rec.append(mem1)
            spk2_rec.append(spk2); mem2_rec.append(mem2)
            spk3_rec.append(spk3); mem3_rec.append(mem3)
            spk4_rec.append(spk4); mem4_rec.append(mem4)

        spk4_t = torch.stack(spk4_rec)   # (T, N, 10)

        recordings = {
            "spk1": torch.stack(spk1_rec), "mem1": torch.stack(mem1_rec),
            "spk2": torch.stack(spk2_rec), "mem2": torch.stack(mem2_rec),
            "spk3": torch.stack(spk3_rec), "mem3": torch.stack(mem3_rec),
            "spk4": spk4_t,               "mem4": torch.stack(mem4_rec),
        }
        return spk4_t, recordings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass returning spike-count logits.

        Args:
            x: (N, C, H, W) image batch with values in [0, 1].

        Returns:
            (N, num_classes) spike-count logits for nn.CrossEntropyLoss.
        """
        spikes = self._encode(x)
        spk_out, _ = self._forward_snn(spikes)
        return spk_out.sum(dim=0)   # (N, num_classes)

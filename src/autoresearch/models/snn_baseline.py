"""SNN Baseline model — Phase 1 of the Artificial Neural Prostheses project.

A feedforward spiking neural network with two hidden fully-connected layers
and an output layer, all using snntorch Leaky Integrate-and-Fire (LIF)
neurons with fast-sigmoid surrogate gradients.

Architecture
------------
Input  : 784  (flattened MNIST pixels, rate-coded over T timesteps)
Hidden1: 512  LIF neurons
Hidden2: 256  LIF neurons
Output : 10   LIF neurons (one per digit class)

The model is designed to be a drop-in replacement for standard PyTorch models:
- ``forward(x)``  accepts an image batch ``(N, C, H, W)`` and returns
  ``(N, num_classes)`` spike-count logits — compatible with the existing
  ``train.py`` / ``validate`` / ``CrossEntropyLoss`` infrastructure.
- ``forward_with_recordings(x)`` returns the full per-layer spike trains and
  membrane potentials for use by ``record.py``.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.spike_encoding import rate_encode, ttfs_encode


class SNNBaseline(nn.Module):
    """Two-hidden-layer feedforward SNN for MNIST classification.

    Args:
        input_size:    Flattened input dimension (default 784 for MNIST).
        hidden1:       Number of neurons in the first hidden layer.
        hidden2:       Number of neurons in the second hidden layer.
        num_classes:   Number of output classes.
        beta:          LIF membrane decay constant (0 < beta < 1).
        threshold:     LIF firing threshold.
        timesteps:     Number of simulation timesteps T for rate coding.
        flatten_input: If True, flatten spatial dimensions before processing.
        encoding:      Spike encoding scheme — 'rate' (Bernoulli) or 'ttfs'.
    """

    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 512,
        hidden2: int = 256,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        timesteps: int = 25,
        flatten_input: bool = True,
        encoding: str = "rate",
    ) -> None:
        super().__init__()

        self.input_size = input_size
        self.timesteps = timesteps
        self.flatten_input = flatten_input
        self.encoding = encoding

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Linear projections
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

        # LIF neurons — one per layer
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image batch to spike trains using the configured scheme.

        Args:
            x: Image batch ``(N, C, H, W)`` or ``(N, input_size)`` with
               pixel values normalised to [0, 1].

        Returns:
            Spike tensor of shape ``(T, N, input_size)``.
        """
        if self.flatten_input:
            x = x.view(x.size(0), -1)  # (N, input_size)
        if self.encoding == "ttfs":
            return ttfs_encode(x, self.timesteps)
        return rate_encode(x, self.timesteps)

    def _forward_snn(
        self, spikes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Run the SNN forward pass over T timesteps.

        Args:
            spikes: Rate-coded input of shape ``(T, N, input_size)``.

        Returns:
            Tuple of:
            - ``spk3_rec`` (T, N, num_classes) — output spike trains
            - ``mem3_rec`` (T, N, num_classes) — output membrane potentials
            - ``recordings`` dict with keys ``spk1``, ``mem1``, ``spk2``,
              ``mem2``, ``spk3``, ``mem3`` each shaped ``(T, N, layer_size)``
        """
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        spk3_rec, mem3_rec = [], []

        for t in range(spikes.shape[0]):
            cur1 = self.fc1(spikes[t])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur3 = self.fc3(spk2)
            spk3, mem3 = self.lif3(cur3, mem3)

            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)
            spk3_rec.append(spk3)
            mem3_rec.append(mem3)

        spk1_t = torch.stack(spk1_rec)  # (T, N, hidden1)
        mem1_t = torch.stack(mem1_rec)
        spk2_t = torch.stack(spk2_rec)  # (T, N, hidden2)
        mem2_t = torch.stack(mem2_rec)
        spk3_t = torch.stack(spk3_rec)  # (T, N, num_classes)
        mem3_t = torch.stack(mem3_rec)

        recordings = {
            "spk1": spk1_t,
            "mem1": mem1_t,
            "spk2": spk2_t,
            "mem2": mem2_t,
            "spk3": spk3_t,
            "mem3": mem3_t,
        }

        return spk3_t, mem3_t, recordings

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard forward pass compatible with the autoresearch train loop.

        Applies Poisson rate coding internally, runs the SNN over T
        timesteps, and returns spike-count logits (sum over T).

        Args:
            x: Image batch ``(N, C, H, W)`` with values in [0, 1].

        Returns:
            Spike-count logits ``(N, num_classes)`` suitable for
            ``nn.CrossEntropyLoss``.
        """
        spikes = self._encode(x)
        spk_out, _, _ = self._forward_snn(spikes)
        return spk_out.sum(dim=0)  # (N, num_classes)

    def forward_with_recordings(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass that also returns per-layer spike trains and potentials.

        Used by ``record.py`` to generate the Phase 1 activation dataset.

        Args:
            x: Image batch ``(N, C, H, W)`` with values in [0, 1].

        Returns:
            Tuple of:
            - ``spk_out``    (T, N, num_classes) — output spike trains
            - ``mem_out``    (T, N, num_classes) — output membrane potentials
            - ``recordings`` dict: keys ``spk1``/``mem1``, ``spk2``/``mem2``,
              ``spk3``/``mem3``, each ``(T, N, layer_size)``
        """
        spikes = self._encode(x)
        return self._forward_snn(spikes)

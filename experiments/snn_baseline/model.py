"""
SNN Baseline — model definition.

A feedforward spiking neural network with two hidden fully-connected layers
and an output layer, all using snntorch Leaky Integrate-and-Fire (LIF) neurons.

Architecture
------------
Input  : 784 (flattened MNIST, repeated over T timesteps via rate coding)
Hidden1: 512 LIF neurons
Hidden2: 256 LIF neurons
Output : 10  LIF neurons (one per digit class)

The network is unrolled over T timesteps. Membrane potentials and spike trains
are recorded for every layer on every forward pass so that record.py can save
them without a second forward pass.
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate


class SNNBaseline(nn.Module):
    """Two-hidden-layer feedforward SNN for MNIST classification."""

    def __init__(
        self,
        input_size: int = 784,
        hidden1: int = 512,
        hidden2: int = 256,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
    ) -> None:
        super().__init__()

        spike_grad = surrogate.fast_sigmoid(slope=25)

        # Linear projections
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)

        # LIF neurons for each layer
        self.lif1 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif2 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)
        self.lif3 = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (T, batch, 784) — rate-coded spike input over T timesteps.

        Returns
        -------
        spk3_rec  : (T, batch, 10)   — output spike trains
        mem3_rec  : (T, batch, 10)   — output membrane potentials
        recordings : dict with keys 'spk1','mem1','spk2','mem2','spk3','mem3'
                     each shaped (T, batch, layer_size)
        """
        T = x.shape[0]
        batch = x.shape[1]

        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()

        spk1_rec, mem1_rec = [], []
        spk2_rec, mem2_rec = [], []
        spk3_rec, mem3_rec = [], []

        for t in range(T):
            cur1 = self.fc1(x[t])
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

        spk1_rec = torch.stack(spk1_rec)  # (T, batch, hidden1)
        mem1_rec = torch.stack(mem1_rec)
        spk2_rec = torch.stack(spk2_rec)  # (T, batch, hidden2)
        mem2_rec = torch.stack(mem2_rec)
        spk3_rec = torch.stack(spk3_rec)  # (T, batch, num_classes)
        mem3_rec = torch.stack(mem3_rec)

        recordings = {
            "spk1": spk1_rec,
            "mem1": mem1_rec,
            "spk2": spk2_rec,
            "mem2": mem2_rec,
            "spk3": spk3_rec,
            "mem3": mem3_rec,
        }

        return spk3_rec, mem3_rec, recordings

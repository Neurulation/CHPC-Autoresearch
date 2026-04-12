"""Spiking Vanilla RNN — Phase B of the ANP SNN project.

Hybrid spiking-recurrent model for sequential MNIST. Identical to SNN-LSTM
but replaces nn.LSTM with nn.RNN (tanh). The input stage uses LIF neurons
(rate coding: T=25 Bernoulli timesteps per row) to produce spike-count features.

Architecture
------------
Input  : (N, 1, 28, 28) MNIST → viewed as 28 rows of 28 pixels
Per row: Bernoulli rate-encode (T=25) → LIF integration → spike counts (28,)
RNN    : 2-layer stacked, input_size=28, hidden_size=256, nonlinearity=tanh
Output : Linear(256→10) logits
"""

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate

from autoresearch.utils.spike_encoding import ttfs_encode


class SNNVanillaRNNModel(nn.Module):
    """Spiking vanilla RNN for sequential MNIST (row-level, T_seq=28).

    Args:
        input_size:      Pixels per row (28 for MNIST).
        hidden_size:     RNN hidden state dimension.
        num_layers:      Number of stacked RNN layers.
        num_classes:     Output classes.
        beta:            LIF membrane decay constant.
        threshold:       LIF firing threshold.
        timesteps:       SNN rate-coding timesteps per row.
        dropout:         RNN inter-layer dropout (ignored if num_layers == 1).
        nonlinearity:    RNN nonlinearity ('tanh' or 'relu').
        encoding:        Spike encoding — 'rate' (Bernoulli) or 'ttfs'.
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 10,
        beta: float = 0.9,
        threshold: float = 1.0,
        timesteps: int = 25,
        dropout: float = 0.3,
        nonlinearity: str = "tanh",
        encoding: str = "rate",
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.timesteps = timesteps
        self.encoding = encoding

        spike_grad = surrogate.fast_sigmoid(slope=25)
        self.lif = snn.Leaky(beta=beta, threshold=threshold, spike_grad=spike_grad)

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            nonlinearity=nonlinearity,
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _encode_row(self, row: torch.Tensor) -> torch.Tensor:
        """Encode a single image row to spike counts via LIF integration.

        Args:
            row: (N, input_size) pixel values in [0, 1].

        Returns:
            (N, input_size) spike counts accumulated over T timesteps.
        """
        row = row.clamp(0.0, 1.0)
        mem = self.lif.init_leaky()
        spk_acc = torch.zeros_like(row)
        if self.encoding == "ttfs":
            spikes_t = ttfs_encode(row, self.timesteps)
        else:
            spikes_t = torch.bernoulli(row.unsqueeze(0).expand(self.timesteps, -1, -1))
        for t in range(self.timesteps):
            spk, mem = self.lif(spikes_t[t], mem)
            spk_acc = spk_acc + spk
        return spk_acc  # (N, input_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (N, C, H, W) MNIST image batch.

        Returns:
            (N, num_classes) logits.
        """
        if x.dim() == 4:
            x = x.squeeze(1)  # (N, 28, 28)

        N, T_seq, _ = x.shape
        encoded = []
        for t in range(T_seq):
            encoded.append(self._encode_row(x[:, t, :]))

        seq = torch.stack(encoded, dim=1)  # (N, T_seq, input_size)
        _, h_n = self.rnn(seq)
        return self.classifier(h_n[-1])  # (N, num_classes)

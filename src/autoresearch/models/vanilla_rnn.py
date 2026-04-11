"""Stacked Vanilla RNN (Elman network) for sequential classification."""

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """Stacked Vanilla RNN (Elman) for sequential image classification.

    Treats each image as a sequence of row vectors.
    For MNIST (1x28x28): T=28 timesteps, input_dim=28.

    Structurally identical to the LSTM and GRU baselines — same architecture
    depth (2 layers), hidden size (256), optimizer (Adam). Uses the simplest
    recurrent cell: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b). No gates,
    no cell state — susceptible to vanishing gradients across long sequences.

    Args:
        input_size: Feature dimension per timestep (e.g. 28 for MNIST rows)
        hidden_size: RNN hidden state dimension
        num_layers: Number of stacked RNN layers
        num_classes: Number of output classes
        dropout: Dropout between RNN layers (ignored if num_layers == 1)
    """

    def __init__(
        self,
        input_size: int = 28,
        hidden_size: int = 256,
        num_layers: int = 2,
        num_classes: int = 10,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        rnn_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=rnn_dropout,
            nonlinearity="tanh",
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, T, input_size)

        Returns:
            Logits of shape (B, num_classes)
        """
        if x.dim() == 4:
            # (B, C, H, W) → (B, H, W) → (B, T=H, input_size=W)
            x = x.squeeze(1)

        _, h_n = self.rnn(x)
        # h_n: (num_layers, B, hidden_size) — take last layer
        return self.classifier(h_n[-1])

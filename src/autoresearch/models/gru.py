"""Stacked GRU for sequential classification."""

import torch
import torch.nn as nn


class GRU(nn.Module):
    """Stacked GRU for sequential image classification.

    Treats each image as a sequence of row vectors.
    For MNIST (1x28x28): T=28 timesteps, input_dim=28.

    Structurally identical to the LSTM baseline but uses GRU cells
    (3 gates vs LSTM's 4, no cell state) — ~25% fewer parameters.

    Args:
        input_size: Feature dimension per timestep (e.g. 28 for MNIST rows)
        hidden_size: GRU hidden state dimension
        num_layers: Number of stacked GRU layers
        num_classes: Number of output classes
        dropout: Dropout between GRU layers (ignored if num_layers == 1)
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

        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
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

        _, h_n = self.gru(x)
        # h_n: (num_layers, B, hidden_size) — take last layer
        return self.classifier(h_n[-1])

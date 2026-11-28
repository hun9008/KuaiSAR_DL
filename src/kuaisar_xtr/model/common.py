from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FieldEmbedding(nn.Module):
    """Embed every categorical field with its own lookup table."""

    def __init__(self, vocab_sizes: Sequence[int], dim: int):
        super().__init__()
        self.tables = nn.ModuleList([nn.Embedding(v, dim, padding_idx=0) for v in vocab_sizes])
        for table in self.tables:
            nn.init.xavier_uniform_(table.weight[1:])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([table(x[:, i]) for i, table in enumerate(self.tables)], dim=1)


def mlp(input_dim: int, hidden_dims: Sequence[int], dropout: float = 0.1) -> nn.Sequential:
    layers: list[nn.Module] = []
    for hidden in hidden_dims:
        layers += [nn.Linear(input_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
        input_dim = hidden
    layers.append(nn.Linear(input_dim, 1))
    return nn.Sequential(*layers)

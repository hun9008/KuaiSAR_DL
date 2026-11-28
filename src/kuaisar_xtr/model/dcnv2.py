from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .common import FieldEmbedding, mlp


class CrossLayer(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.weight = nn.Linear(width, width, bias=False)
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.weight(x) + self.bias + x


class DCNv2(nn.Module):
    def __init__(self, vocab_sizes: Sequence[int], dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.embedding = FieldEmbedding(vocab_sizes, dim)
        width = len(vocab_sizes) * dim
        self.cross = nn.ModuleList([CrossLayer(width) for _ in range(3)])
        self.deep = mlp(width, hidden_dims[:-1] or (128,))
        self.output = nn.Linear(width + 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding(x).flatten(1)
        crossed = x0
        for layer in self.cross:
            crossed = layer(x0, crossed)
        deep_logit = self.deep(x0)
        return self.output(torch.cat([crossed, deep_logit], dim=1)).squeeze(1)

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .common import FieldEmbedding, mlp


class DeepFM(nn.Module):
    def __init__(self, vocab_sizes: Sequence[int], dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        self.embedding = FieldEmbedding(vocab_sizes, dim)
        self.linear = FieldEmbedding(vocab_sizes, 1)
        self.deep = mlp(len(vocab_sizes) * dim, hidden_dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)
        linear = self.linear(x).sum((1, 2))
        fm = 0.5 * ((emb.sum(1) ** 2 - (emb**2).sum(1)).sum(1))
        return linear + fm + self.deep(emb.flatten(1)).squeeze(1)

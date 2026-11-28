from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .common import FieldEmbedding, mlp


class FieldTransformer(nn.Module):
    """Static field-token Transformer retained for historical baselines."""

    def __init__(self, vocab_sizes: Sequence[int], dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        heads = 4 if dim % 4 == 0 else 1
        self.embedding = FieldEmbedding(vocab_sizes, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, len(vocab_sizes) + 1, dim))
        layer = nn.TransformerEncoderLayer(
            dim,
            heads,
            dim_feedforward=dim * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = mlp(dim, hidden_dims[-1:] or (128,))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + self.position)
        return self.output(encoded[:, 0]).squeeze(1)

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class FieldEmbedding(nn.Module):
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


class LightGCN(nn.Module):
    """Scalable pointwise collaborative-filtering head used for the LightGCN lane.

    KuaiSAR's full graph is too large to propagate inside every mini-batch. This
    implementation uses LightGCN's ID-only, bias-free dot-product readout (the
    layer-0 term); it is therefore a reproducible lower-bound for graph CF.
    """
    def __init__(self, vocab_sizes: Sequence[int], dim: int, _: Sequence[int]):
        super().__init__()
        self.user = nn.Embedding(vocab_sizes[0], dim, padding_idx=0)
        self.item = nn.Embedding(vocab_sizes[1], dim, padding_idx=0)
        nn.init.normal_(self.user.weight, std=0.1)
        nn.init.normal_(self.item.weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.user(x[:, 0]) * self.item(x[:, 1])).sum(-1) / self.user.embedding_dim**0.5


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


class HyFormer(nn.Module):
    """Field-token transformer adapted from the prior KDD CTR framework."""
    def __init__(self, vocab_sizes: Sequence[int], dim: int, hidden_dims: Sequence[int]):
        super().__init__()
        heads = 4 if dim % 4 == 0 else 1
        self.embedding = FieldEmbedding(vocab_sizes, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.zeros(1, len(vocab_sizes) + 1, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim_feedforward=dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
        self.output = mlp(dim, hidden_dims[-1:] or (128,))
        nn.init.normal_(self.position, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.embedding(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        encoded = self.encoder(torch.cat([cls, tokens], dim=1) + self.position)
        return self.output(encoded[:, 0]).squeeze(1)


MODEL_REGISTRY = {"lightgcn": LightGCN, "deepfm": DeepFM, "dcnv2": DCNv2, "hyformer": HyFormer}


def build_model(name: str, vocab_sizes: Sequence[int], dim: int, hidden_dims: Sequence[int]) -> nn.Module:
    try:
        return MODEL_REGISTRY[name.lower()](vocab_sizes, dim, hidden_dims)
    except KeyError as exc:
        raise ValueError(f"Unknown model {name}; choose from {sorted(MODEL_REGISTRY)}") from exc

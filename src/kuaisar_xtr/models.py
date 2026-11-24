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
    """Standard LightGCN propagation over a fixed train-only bipartite graph."""

    def __init__(
        self,
        vocab_sizes: Sequence[int],
        dim: int,
        _: Sequence[int],
        edge_users: torch.Tensor | None = None,
        edge_items: torch.Tensor | None = None,
        num_layers: int = 2,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        if self.num_layers < 1:
            raise ValueError("LightGCN requires at least one propagation layer")
        self.user = nn.Embedding(vocab_sizes[0], dim, padding_idx=0)
        self.item = nn.Embedding(vocab_sizes[1], dim, padding_idx=0)
        nn.init.normal_(self.user.weight, std=0.02)
        nn.init.normal_(self.item.weight, std=0.02)
        with torch.no_grad():
            self.user.weight[0].zero_()
            self.item.weight[0].zero_()

        edge_users = torch.empty(0, dtype=torch.long) if edge_users is None else edge_users.long()
        edge_items = torch.empty(0, dtype=torch.long) if edge_items is None else edge_items.long()
        if edge_users.shape != edge_items.shape:
            raise ValueError("LightGCN user/item edge tensors must have identical shapes")
        if edge_users.numel() and (
            edge_users.min() < 1 or edge_users.max() >= vocab_sizes[0]
            or edge_items.min() < 1 or edge_items.max() >= vocab_sizes[1]
        ):
            raise ValueError("LightGCN edge ID is outside the embedding vocabulary")
        self.register_buffer("edge_users", edge_users)
        self.register_buffer("edge_items", edge_items)
        user_degree = torch.bincount(edge_users, minlength=vocab_sizes[0]).float().clamp_min(1).pow(-0.5)
        item_degree = torch.bincount(edge_items, minlength=vocab_sizes[1]).float().clamp_min(1).pow(-0.5)
        self.register_buffer("user_degree_inv_sqrt", user_degree)
        self.register_buffer("item_degree_inv_sqrt", item_degree)

    def propagate(self) -> tuple[torch.Tensor, torch.Tensor]:
        user, item = self.user.weight, self.item.weight
        users, items = [user], [item]
        if not self.edge_users.numel():
            return user, item
        norm = self.user_degree_inv_sqrt[self.edge_users] * self.item_degree_inv_sqrt[self.edge_items]
        for _ in range(self.num_layers):
            next_user = torch.zeros_like(user)
            next_item = torch.zeros_like(item)
            next_user.index_add_(0, self.edge_users, item[self.edge_items] * norm.unsqueeze(1))
            next_item.index_add_(0, self.edge_items, user[self.edge_users] * norm.unsqueeze(1))
            user, item = next_user, next_item
            users.append(user)
            items.append(item)
        return torch.stack(users).mean(0), torch.stack(items).mean(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        user, item = self.propagate()
        return (user[x[:, 0]] * item[x[:, 1]]).sum(-1)


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


def build_model(
    name: str,
    vocab_sizes: Sequence[int],
    dim: int,
    hidden_dims: Sequence[int],
    *,
    edge_users: torch.Tensor | None = None,
    edge_items: torch.Tensor | None = None,
    lightgcn_layers: int = 2,
) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name}; choose from {sorted(MODEL_REGISTRY)}") from exc
    if model_class is LightGCN:
        return model_class(vocab_sizes, dim, hidden_dims, edge_users, edge_items, lightgcn_layers)
    return model_class(vocab_sizes, dim, hidden_dims)

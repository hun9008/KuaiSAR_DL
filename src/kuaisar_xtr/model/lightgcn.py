from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


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
            edge_users.min() < 1
            or edge_users.max() >= vocab_sizes[0]
            or edge_items.min() < 1
            or edge_items.max() >= vocab_sizes[1]
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

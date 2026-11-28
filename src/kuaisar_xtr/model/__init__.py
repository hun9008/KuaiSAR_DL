from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from torch import nn

from .common import FieldEmbedding
from .dcnv2 import DCNv2
from .deepfm import DeepFM
from .field_transformer import FieldTransformer
from .hyformer import ModelInput, PCVRHyFormer
from .lightgcn import LightGCN

# Public project name while preserving the original KDD class and state-dict
# layout in ``hyformer.py``.
HyFormer = PCVRHyFormer

MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "lightgcn": LightGCN,
    "deepfm": DeepFM,
    "dcnv2": DCNv2,
    "field_transformer": FieldTransformer,
    "hyformer": HyFormer,
}


def build_model(
    name: str,
    vocab_sizes: Sequence[int],
    dim: int,
    hidden_dims: Sequence[int],
    *,
    edge_users: torch.Tensor | None = None,
    edge_items: torch.Tensor | None = None,
    lightgcn_layers: int = 2,
    hyformer_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    try:
        model_class = MODEL_REGISTRY[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown model {name}; choose from {sorted(MODEL_REGISTRY)}") from exc
    if model_class is LightGCN:
        return model_class(vocab_sizes, dim, hidden_dims, edge_users, edge_items, lightgcn_layers)
    if model_class is HyFormer:
        if hyformer_kwargs is None:
            raise ValueError(
                "HyFormer requires sequence schema/model arguments via hyformer_kwargs; "
                "use field_transformer for the former static baseline"
            )
        return model_class(**dict(hyformer_kwargs))
    return model_class(vocab_sizes, dim, hidden_dims)


__all__ = [
    "DCNv2",
    "DeepFM",
    "FieldEmbedding",
    "FieldTransformer",
    "HyFormer",
    "LightGCN",
    "MODEL_REGISTRY",
    "ModelInput",
    "PCVRHyFormer",
    "build_model",
]

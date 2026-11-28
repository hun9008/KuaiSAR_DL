"""Backward-compatible model imports.

New code should import from :mod:`kuaisar_xtr.model`.
"""

from .model import (
    DCNv2,
    DeepFM,
    FieldEmbedding,
    FieldTransformer,
    HyFormer,
    LightGCN,
    MODEL_REGISTRY,
    ModelInput,
    PCVRHyFormer,
    build_model,
)

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

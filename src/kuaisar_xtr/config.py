from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ExperimentConfig:
    model: str
    task: str
    regime: str = "original"
    data_dir: str = "data/processed"
    output_dir: str = "runs"
    seed: int = 42
    batch_size: int = 4096
    epochs: int = 10
    patience: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    embedding_dim: int = 32
    lightgcn_layers: int = 2
    lightgcn_batch_size: int = 65_536
    hidden_dims: tuple[int, ...] = (256, 128)
    num_workers: int = 2
    device: str = "auto"
    max_train_steps: int | None = None

    @classmethod
    def from_json(cls, path: str | Path) -> "ExperimentConfig":
        payload: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        if "hidden_dims" in payload:
            payload["hidden_dims"] = tuple(payload["hidden_dims"])
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["hidden_dims"] = list(self.hidden_dims)
        return result

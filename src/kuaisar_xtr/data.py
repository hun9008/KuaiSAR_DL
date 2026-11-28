from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch
from torch.utils.data import IterableDataset, get_worker_info


class ParquetBatchDataset(IterableDataset):
    """Row-group sharded iterable dataset that does not load the corpus in RAM."""

    def __init__(
        self,
        path: str | Path,
        features: list[str],
        label: str,
        batch_size: int,
        shuffle: bool,
        seed: int,
        drop_singleton: bool = False,
    ):
        super().__init__()
        self.path, self.features, self.label = Path(path), features, label
        self.batch_size, self.shuffle, self.seed = batch_size, shuffle, seed
        self.drop_singleton = drop_singleton
        self.epoch = 0
        parquet = pq.ParquetFile(self.path)
        self.row_groups = list(range(parquet.metadata.num_row_groups))
        self.num_rows = parquet.metadata.num_rows

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return math.ceil(self.num_rows / self.batch_size)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        worker = get_worker_info()
        groups = self.row_groups.copy()
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(groups)
        if worker:
            groups = groups[worker.id :: worker.num_workers]
        parquet = pq.ParquetFile(self.path)
        columns = self.features + [self.label]
        # Unknown labels are dropped before batching and rows carry over across row
        # groups, so sparse labels (END30 on the original split) still form full
        # batch_size batches instead of tiny, sometimes single-row, ones.
        pending: pa.Table | None = None
        for group in groups:
            table = parquet.read_row_group(group, columns=columns)
            table = table.filter(pc.is_valid(table[self.label]))
            if table.num_rows == 0:
                continue
            if self.shuffle:
                # Row-group order alone keeps on-disk ordering inside each batch.
                rng = np.random.default_rng([self.seed, self.epoch, group])
                table = table.take(rng.permutation(table.num_rows))
            pending = table if pending is None else pa.concat_tables([pending, table])
            while pending.num_rows >= self.batch_size:
                yield self._to_tensors(pending.slice(0, self.batch_size))
                pending = pending.slice(self.batch_size)
        # A single-row training batch breaks BatchNorm; evaluation keeps every row.
        if pending is not None and pending.num_rows and (not self.shuffle or pending.num_rows > 1):
            yield self._to_tensors(pending)

    def _to_tensors(self, table: pa.Table) -> tuple[torch.Tensor, torch.Tensor]:
        frame = table.to_pandas()
        x = frame[self.features].to_numpy(dtype=np.int64, copy=True)
        y = frame[self.label].to_numpy(dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_data_spec(data_dir: str | Path) -> tuple[list[str], list[int]]:
    manifest = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    features = manifest["features"]
    vocab_sizes = [int(manifest["vocab_sizes"][name]) + 1 for name in features]
    return features, vocab_sizes


def load_lightgcn_graph(data_dir: str | Path, task: str) -> tuple[torch.Tensor, torch.Tensor, dict]:
    root = Path(data_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    graph_spec = manifest.get("lightgcn_graphs")
    if not graph_spec:
        raise ValueError("Dataset has no LightGCN graph metadata; rerun data preparation")
    try:
        action = graph_spec["task_to_action"][task]
        action_spec = graph_spec["actions"][action]
    except KeyError as exc:
        raise ValueError(f"No LightGCN graph is defined for task {task}") from exc
    path = root / action_spec["path"]
    table = pq.read_table(path, columns=["user_id", "item_id"])
    users = torch.from_numpy(table["user_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=True))
    items = torch.from_numpy(table["item_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=True))
    return users, items, {"action": action, "path": str(path), "edges": int(len(users))}


def split_path(data_dir: str | Path, regime: str, task: str, split: str) -> Path:
    root = Path(data_dir)
    if regime == "sampled_1_10" and split == "train":
        return root / regime / task / "train.parquet"
    if regime not in {"original", "sampled_1_10"}:
        raise ValueError(f"Unknown regime: {regime}")
    return root / "original" / f"{split}.parquet"

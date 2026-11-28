from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from kuaisar_xtr.data import ParquetBatchDataset, split_path


def test_parquet_dataset_filters_unknown_labels(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"a": [1, 2, 3], "b": [4, 5, 6], "like_end_30m": [1, None, 0]}), path)
    dataset = ParquetBatchDataset(path, ["a", "b"], "like_end_30m", batch_size=8, shuffle=False, seed=1)
    x, y = next(iter(dataset))
    assert x.shape == (2, 2)
    assert y.tolist() == [1.0, 0.0]


def test_sampled_regime_only_changes_train_path(tmp_path: Path) -> None:
    assert split_path(tmp_path, "sampled_1_10", "like", "train") == tmp_path / "sampled_1_10/like/train.parquet"
    assert split_path(tmp_path, "sampled_1_10", "like", "valid") == tmp_path / "original/valid.parquet"


def test_train_shuffle_mixes_rows_within_row_group(tmp_path: Path) -> None:
    # Label-sorted file with a single row group mimics the class-partitioned sampled output.
    path = tmp_path / "sorted.parquet"
    labels = [0] * 900 + [1] * 100
    pq.write_table(pa.table({"a": list(range(1000)), "forward": labels}), path, row_group_size=1000)
    dataset = ParquetBatchDataset(path, ["a"], "forward", batch_size=100, shuffle=True, seed=42)
    batches = [y for _, y in dataset]
    assert sum(float(y.sum()) for y in batches) == 100
    assert max(float(y.mean()) for y in batches) < 1.0
    assert [y.tolist() for _, y in dataset] == [y.tolist() for y in batches]  # deterministic per epoch
    unshuffled = [y for _, y in ParquetBatchDataset(path, ["a"], "forward", 100, shuffle=False, seed=42)]
    assert float(unshuffled[-1].mean()) == 1.0

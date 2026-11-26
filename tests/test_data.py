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


def test_sparse_labels_fill_full_batches_across_row_groups(tmp_path: Path) -> None:
    # END30-like column: only every 10th row is labeled, split over several row groups.
    path = tmp_path / "sparse.parquet"
    labels = [(i // 10) % 2 if i % 10 == 0 else None for i in range(1010)]
    pq.write_table(pa.table({"a": list(range(1010)), "forward_end_30m": labels}), path, row_group_size=100)
    train = [y for _, y in ParquetBatchDataset(path, ["a"], "forward_end_30m", 8, shuffle=True, seed=1)]
    assert [len(y) for y in train] == [8] * 12 + [5]  # 101 labeled rows, carried over row groups
    evaluation = [y for _, y in ParquetBatchDataset(path, ["a"], "forward_end_30m", 50, shuffle=False, seed=1)]
    assert [len(y) for y in evaluation] == [50, 50, 1]  # evaluation keeps every labeled row


def test_training_drops_single_row_remainder(tmp_path: Path) -> None:
    path = tmp_path / "remainder.parquet"
    pq.write_table(pa.table({"a": list(range(9)), "forward": [0, 1] * 4 + [1]}), path)
    train = [y for _, y in ParquetBatchDataset(path, ["a"], "forward", 4, shuffle=True, seed=1)]
    assert [len(y) for y in train] == [4, 4]

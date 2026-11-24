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

from pathlib import Path

import pandas as pd

from kuaisar_xtr.prepare import prepare_dataset
from kuaisar_xtr.config import ExperimentConfig
from kuaisar_xtr.trainer import run_experiment


def test_prepare_dataset_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    users = pd.DataFrame({
        "user_id": [1, 2, 3, 4], "onehot_feat1": [1, 2, 1, 2],
        "onehot_feat2": [2, 1, 2, 1], "search_active_level": [1, 2, 3, 4],
        "rec_active_level": [4, 3, 2, 1],
    })
    items = pd.DataFrame({
        "item_id": list(range(1, 13)),
        "first_level_category_name_en": ["a", "b"] * 6,
        "second_level_category_name_en": ["x", "y", "z"] * 4,
        "item_type": ["photo", "video"] * 6, "upload_type": ["normal"] * 12,
    })
    rows = []
    base = pd.Timestamp("2026-01-01")
    for index in range(80):
        # Per-user next-event gaps alternate across/below the 30-minute boundary.
        gap = 2_400 if index % 8 == 0 else 30
        rows.append({
            "user_id": index % 4 + 1, "item_id": index % 12 + 1,
            "time": base + pd.Timedelta(seconds=sum(2_400 if j % 8 == 0 else 30 for j in range(index + 1))),
            "like": index % 5 == 0, "search": index % 7 == 0,
            "follow": index % 9 == 0, "forward": index % 11 == 0,
            "duration_ms": gap * 1000, "playing_time": gap * 500,
        })
    users.to_csv(raw / "user_features.csv", index=False)
    items.to_csv(raw / "item_features.csv", index=False)
    pd.DataFrame(rows).to_csv(raw / "rec_inter.csv", index=False)
    output = tmp_path / "processed"
    manifest = prepare_dataset(raw, output)
    assert (output / "original/train.parquet").exists()
    assert (output / "sampled_1_10/like/train.parquet").exists()
    assert manifest["sampling"]["train_only"] is True
    assert set(manifest["tasks"]) == {
        "like", "search", "follow", "forward", "like_end_30m",
        "search_end_30m", "follow_end_30m", "forward_end_30m",
    }
    original = pd.read_parquet(output / "original/train.parquet")
    known_end = original["like_end_30m"].notna()
    assert known_end.any()
    assert (original.loc[known_end, "like"] == 1).all()
    assert original.loc[known_end, "like_end_30m"].nunique() == 2
    sampled = pd.read_parquet(output / "sampled_1_10/like/train.parquet")
    positives = int(sampled["like"].sum())
    assert len(sampled) - positives <= positives * 10
    result = run_experiment(ExperimentConfig(
        model="lightgcn", task="like", data_dir=str(output),
        output_dir=str(tmp_path / "runs"), batch_size=16, epochs=1,
        patience=1, embedding_dim=4, hidden_dims=(8,), num_workers=0,
        device="cpu", max_train_steps=2,
    ))
    assert result["test_rows"] > 0
    assert (tmp_path / "runs/lightgcn__like__original__seed42/metrics.json").exists()

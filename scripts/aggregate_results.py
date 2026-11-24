#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate metrics.json files")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output", default="runs/summary.csv")
    args = parser.parse_args()
    records = [json.loads(path.read_text(encoding="utf-8")) for path in Path(args.runs_dir).glob("*/metrics.json")]
    if not records:
        raise SystemExit(f"No metrics.json under {args.runs_dir}")
    frame = pd.DataFrame(records).sort_values(["task", "regime", "model", "seed"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    aggregate = (
        frame.groupby(["task", "regime", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            test_auc_mean=("test_auc", "mean"),
            test_auc_std=("test_auc", "std"),
            test_logloss_mean=("test_logloss", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
        .sort_values(["task", "regime", "test_auc_mean"], ascending=[True, True, False])
    )
    aggregate_path = output.with_name(output.stem + "_aggregate" + output.suffix)
    aggregate.to_csv(aggregate_path, index=False)
    print(frame.to_string(index=False))
    print(f"\nAggregated metrics: {aggregate_path}")


if __name__ == "__main__":
    main()

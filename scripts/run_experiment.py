#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuaisar_xtr.config import ExperimentConfig
from kuaisar_xtr.trainer import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one KuaiSAR XTR experiment")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", choices=["lightgcn", "deepfm", "dcnv2", "hyformer"])
    parser.add_argument("--task")
    parser.add_argument("--regime", choices=["original", "sampled_1_10"])
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-train-steps", type=int)
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload.pop("models", None)
    payload.pop("tasks", None)
    payload.pop("regimes", None)
    for name in ("model", "task", "regime", "device", "seed", "max_train_steps"):
        value = getattr(args, name)
        if value is not None:
            payload[name] = value
    if "task" not in payload:
        parser.error("--task is required when the config is an assignment matrix")
    if "model" not in payload:
        parser.error("--model is required when the config is an assignment matrix")
    config = ExperimentConfig(**payload)
    print(json.dumps(run_experiment(config), indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuaisar_xtr.config import ExperimentConfig
from kuaisar_xtr.trainer import run_experiment
from kuaisar_xtr.utils import atomic_json_dump


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one member's task/regime matrix")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    parser.add_argument("--resume", action="store_true", help="Skip runs with metrics.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    models, tasks, regimes = payload.pop("models"), payload.pop("tasks"), payload.pop("regimes")
    assignment_name = "__".join(tasks)
    results = []
    total = len(tasks) * len(models) * len(regimes)
    run_index = 0
    print(f"[ASSIGNMENT] {assignment_name}: {total} runs", flush=True)
    for task in tasks:
        for model in models:
            for regime in regimes:
                run_index += 1
                config = ExperimentConfig(model=model, task=task, regime=regime, **payload)
                if args.device:
                    config.device = args.device
                run_name = f"{model}__{task}__{regime}__seed{config.seed}"
                metrics = Path(config.output_dir) / run_name / "metrics.json"
                if args.resume and metrics.exists():
                    print(f"[{run_index}/{total}] SKIP completed {run_name}", flush=True)
                    results.append(json.loads(metrics.read_text(encoding="utf-8")))
                else:
                    print(f"[{run_index}/{total}] RUN {run_name}", flush=True)
                    results.append(run_experiment(config))
                atomic_json_dump(results, Path(config.output_dir) / f"assignment_{assignment_name}.json")
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()

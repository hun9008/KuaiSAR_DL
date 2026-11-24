#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from kuaisar_xtr.prepare import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KuaiSAR XTR datasets")
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--negative-ratio", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    manifest = prepare_dataset(**vars(args))
    print(json.dumps(manifest["stats"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
set -euo pipefail
umask 002

python_bin="${PYTHON_BIN:-python}"
if [[ -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 {download|prepare|member1|member2|member3|member4|aggregate} [args...]" >&2
  exit 2
fi

command_name="$1"
shift
case "$command_name" in
  download) scripts/download_data.sh "$@" ;;
  prepare) "$python_bin" scripts/prepare_data.py "$@" ;;
  member1) "$python_bin" scripts/run_assignment.py --config configs/member1_like.json "$@" ;;
  member2) "$python_bin" scripts/run_assignment.py --config configs/member2_search.json "$@" ;;
  member3) "$python_bin" scripts/run_assignment.py --config configs/member3_follow.json "$@" ;;
  member4) "$python_bin" scripts/run_assignment.py --config configs/member4_share.json "$@" ;;
  aggregate) "$python_bin" scripts/aggregate_results.py "$@" ;;
  *) echo "Unknown command: $command_name" >&2; exit 2 ;;
esac

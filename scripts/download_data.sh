#!/usr/bin/env bash
set -euo pipefail

archive_dir="${1:-data/downloads}"
raw_dir="${2:-data/raw}"
archive="${archive_dir}/KuaiSAR_v2.zip"
url="https://zenodo.org/records/8181109/files/KuaiSAR_v2.zip"
expected_md5="ed630f47fb013c5b31bb020c483ef262"

mkdir -p "$archive_dir" "$raw_dir"
if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 16 -s 16 -k 1M --file-allocation=none \
    -d "$archive_dir" -o KuaiSAR_v2.zip "$url"
else
  wget -c -O "$archive" "$url"
fi

actual_md5="$(md5sum "$archive" | awk '{print $1}')"
if [[ "$actual_md5" != "$expected_md5" ]]; then
  echo "MD5 mismatch: expected=$expected_md5 actual=$actual_md5" >&2
  exit 1
fi

unzip -q -o "$archive" -d "$raw_dir"
echo "Ready: ${raw_dir}/KuaiSAR_v2"

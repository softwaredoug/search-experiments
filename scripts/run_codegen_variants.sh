#!/usr/bin/env bash
set -euo pipefail

dataset="${1:-wands}"
if [ $# -gt 0 ]; then
  shift
fi

configs=(
  "configs/codegen/codegen_no_guards.yml"
  "configs/codegen/codegen_guarded.yml"
)

for config in "${configs[@]}"; do
  echo "Running: ${config} on dataset=${dataset}"
  uv run run --strategy "${config}" --dataset "${dataset}" "$@"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/wands_grep_stats.csv"

mkdir -p "$(dirname "${RESULTS_CSV}")"

CONFIGS=(
  "configs/ecom_base/bm25.yml"
  "configs/ecom_base/embedding_e5_base_v2.yml"
  "configs/wands_grep/agentic_wands_fs_tools.yml"
  "configs/wands_grep/agentic_wands_bash.yml"
  "configs/wands_grep/agentic_wands_bash_delegate.yml"
  "configs/wands_grep/agentic_wands_bash_delegate_todos.yml"
)

for config in "${CONFIGS[@]}"; do
  uv run run \
    --strategy "${ROOT_DIR}/${config}" \
    --dataset "wands" \
    --workers 4 \
    --device mps \
    --summary-csv "${RESULTS_CSV}" \
    --no-cache
done

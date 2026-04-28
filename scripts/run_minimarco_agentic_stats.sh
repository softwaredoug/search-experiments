#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/results_minimarco.csv"

CONFIGS=(
  "configs/msmarco/agentic_msmarco_e5_gpt5_mini.yml"
  "configs/msmarco/agentic_msmarco_bm25_gpt5_mini.yml"
  "configs/msmarco/agentic_msmarco_bm25_e5_gpt5_mini.yml"
  "configs/msmarco/bm25_msmarco.yml"
  "configs/msmarco/embedding_e5_msmarco.yml"
)

DATASET="minimarco"

for config in "${CONFIGS[@]}"; do
  uv run run \
    --strategy "${ROOT_DIR}/${config}" \
    --dataset "${DATASET}" \
    --workers 16 \
    --device mps \
    --summary-csv "${RESULTS_CSV}"
done

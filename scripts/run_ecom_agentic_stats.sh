#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/results.csv"

CONFIGS=(
  "configs/agentic_ecom_bm25_gpt5.yml"
  "configs/agentic_ecom_embedding_gpt5.yml"
  "configs/agentic_ecom_2tools_gpt5.yml"
  "configs/agentic_ecom_bm25_gpt5_mini.yml"
  "configs/agentic_ecom_embedding_gpt5_mini.yml"
  "configs/agentic_ecom_2tools_gpt5_mini.yml"
  "configs/bm25.yml"
  "configs/embedding_minilm.yml"
)

DATASETS=(
  "esci"
  "wands"
)

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    if [[ "${dataset}" == "esci" ]]; then
      uv run run \
        --strategy "${ROOT_DIR}/${config}" \
        --dataset "${dataset}" \
        --num-queries 1000 \
        --workers 16 \
        --device mps \
        --summary-csv "${RESULTS_CSV}"
    else
      uv run run \
        --strategy "${ROOT_DIR}/${config}" \
        --dataset "${dataset}" \
        --workers 16 \
        --device mps \
        --summary-csv "${RESULTS_CSV}"
    fi
  done
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/results.csv"

CONFIGS=(
  "configs/ecom_mincalls/agentic_ecom_2tools_4calls_repeat_gpt5_mini.yml"
  "configs/ecom_mincalls/agentic_ecom_2tools_4calls_sim0p9_gpt5_mini.yml"
  "configs/ecom_mincalls/agentic_ecom_2tools_6calls_repeat_gpt5_mini.yml"
  "configs/ecom_mincalls/agentic_ecom_2tools_6calls_sim0p9_gpt5_mini.yml"
  "configs/ecom_mincalls/agentic_ecom_2tools_8calls_repeat_gpt5_mini.yml"
  "configs/ecom_mincalls/agentic_ecom_2tools_8calls_sim0p9_gpt5_mini.yml"
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

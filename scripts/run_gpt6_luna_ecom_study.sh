#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/gpt6_luna_ecom_study.csv}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/gpt6_luna_ecom_study}"
WORKERS="${WORKERS:-8}"
MODEL_WORKERS="${MODEL_WORKERS:-24}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
NO_CACHE="${NO_CACHE:-false}"
ESCI_NUM_QUERIES="${ESCI_NUM_QUERIES:-1000}"
WANDS_NUM_QUERIES="${WANDS_NUM_QUERIES:-}"

CONFIGS=(
  "configs/agentic/gpt-6-luna-study/bm25.yml"
  "configs/agentic/gpt-6-luna-study/agentic_gpt5_mini_bm25.yml"
  "configs/agentic/gpt-6-luna-study/agentic_gpt5_mini_bm25_e5.yml"
  "configs/agentic/gpt-6-luna-study/agentic_gpt6_luna_bm25.yml"
  "configs/agentic/gpt-6-luna-study/agentic_gpt6_luna_bm25_e5.yml"
)
DATASETS=("wands" "esci")

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT_PREFIX}")"
: > "${RESULTS_CSV}"
cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    workers="${WORKERS}"
    if [[ "${config}" == *gpt* ]]; then
      workers="${MODEL_WORKERS}"
    fi

    echo "Running ${config} on ${dataset} with ${workers} worker(s)"
    run_args=(
      uv run run
      --strategy "${ROOT_DIR}/${config}"
      --dataset "${dataset}"
      --workers "${workers}"
      --seed "${SEED}"
      --summary-csv "${RESULTS_CSV}"
    )
    if [[ "${NO_CACHE}" == "true" ]]; then
      run_args+=(--no-cache)
    fi
    if [[ -n "${DEVICE}" ]]; then
      run_args+=(--device "${DEVICE}")
    fi
    if [[ "${dataset}" == "esci" ]]; then
      run_args+=(--num-queries "${ESCI_NUM_QUERIES}")
    elif [[ "${dataset}" == "wands" && -n "${WANDS_NUM_QUERIES}" ]]; then
      run_args+=(--num-queries "${WANDS_NUM_QUERIES}")
    fi

    "${run_args[@]}"
  done
done

uv run python "${ROOT_DIR}/scripts/plot_gpt6_luna_ecom_study.py" \
  --input "${RESULTS_CSV}" \
  --output-prefix "${PLOT_OUTPUT_PREFIX}"

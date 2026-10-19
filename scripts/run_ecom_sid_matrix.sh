#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/ecom_sid_matrix.csv"
PLOT_OUTPUT="${ROOT_DIR}/assets/ecom_sid_matrix.png"
PARETO_OUTPUT="${ROOT_DIR}/assets/ecom_sid_matrix_pareto.png"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
NO_CACHE="${NO_CACHE:-false}"
ESCI_NUM_QUERIES="${ESCI_NUM_QUERIES:-1000}"
WANDS_NUM_QUERIES="${WANDS_NUM_QUERIES:-}"

CONFIGS=(
  "configs/ecom_base/bm25.yml"
  "configs/agentic_ecom_bm25_gpt5.yml"
  "configs/ecom_agentic/agentic_ecom_2tools_gpt5.yml"
  "configs/ecom_agentic/agentic_ecom_bm25_gpt5_mini.yml"
  "configs/ecom_agentic/agentic_ecom_2tools_e5_gpt5_mini.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_low.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_medium.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_high.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_low.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_medium.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_high.yml"
)
DATASETS=("wands" "esci")

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT}")"
rm -f "${RESULTS_CSV}"

cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    strategy="${ROOT_DIR}/${config}"

    echo "Running ${config} on ${dataset} with ${WORKERS} worker(s)"
    run_args=(
      uv run run
      --strategy "${strategy}"
      --dataset "${dataset}"
      --workers "${WORKERS}"
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

uv run python "${ROOT_DIR}/scripts/plot_ecom_sid_matrix.py" \
  --input "${RESULTS_CSV}" \
  --output "${PLOT_OUTPUT}" \
  --pareto-output "${PARETO_OUTPUT}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${PLOT_OUTPUT}"
echo "Wrote ${PARETO_OUTPUT}"

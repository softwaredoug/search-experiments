#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/ecom_decisions_variants.csv}"
PLOT_OUTPUT="${PLOT_OUTPUT:-${ROOT_DIR}/assets/ecom_decisions_variants.png}"
WORKERS="${WORKERS:-16}"
SEED="${SEED:-42}"
WANDS_NUM_QUERIES="${WANDS_NUM_QUERIES:-480}"
ESCI_NUM_QUERIES="${ESCI_NUM_QUERIES:-1000}"

CONFIGS=(
  "configs/ecom_base/bm25.yml"
  "configs/ecom_decisions/bag_of_decisions_direct.yml"
  "configs/ecom_decisions/bag_of_decisions.yml"
)

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT}")"
: > "${RESULTS_CSV}"
cd "${ROOT_DIR}"

for dataset in wands esci; do
  case "${dataset}" in
    wands) num_queries="${WANDS_NUM_QUERIES}" ;;
    esci) num_queries="${ESCI_NUM_QUERIES}" ;;
  esac

  for config in "${CONFIGS[@]}"; do
    echo "Running ${config} on ${dataset} (${num_queries} queries)"
    uv run run \
      --strategy "${ROOT_DIR}/${config}" \
      --dataset "${dataset}" \
      --num-queries "${num_queries}" \
      --seed "${SEED}" \
      --workers "${WORKERS}" \
      --summary-csv "${RESULTS_CSV}"
  done
done

uv run python "${SCRIPT_DIR}/plot_ecom_decisions_variants.py" \
  --input "${RESULTS_CSV}" \
  --output "${PLOT_OUTPUT}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${PLOT_OUTPUT}"

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/wands_llm_as_judge_variants.csv"
PLOT_OUTPUT="${ROOT_DIR}/assets/wands_llm_as_judge_agreement.png"
WORKERS="${WORKERS:-1}"
SEED="${SEED:-42}"
NUM_QUERIES="${NUM_QUERIES:-}"
NO_CACHE="${NO_CACHE:-false}"

CONFIGS=(
  "configs/ecom_llm_as_judge/wands_pointwise_judge_gpt5_mini.yml"
  "configs/ecom_llm_as_judge/wands_pointwise_judge_gpt5.yml"
  "configs/ecom_llm_as_judge/wands_pointwise_judge_jev_latest.yml"
  "configs/ecom_llm_as_judge/wands_pairwise_judge_gpt5_mini.yml"
  "configs/ecom_llm_as_judge/wands_pairwise_judge_jev_latest.yml"
)

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT}")"
rm -f "${RESULTS_CSV}"

cd "${ROOT_DIR}"

for config in "${CONFIGS[@]}"; do
  strategy="${ROOT_DIR}/${config}"
  echo "Running ${config} on WANDS"

  run_args=(
    uv run run
    --strategy "${strategy}"
    --dataset wands
    --workers "${WORKERS}"
    --seed "${SEED}"
    --summary-csv "${RESULTS_CSV}"
  )
  if [[ -n "${NUM_QUERIES}" ]]; then
    run_args+=(--num-queries "${NUM_QUERIES}")
  fi
  if [[ "${NO_CACHE}" == "true" ]]; then
    run_args+=(--no-cache)
  fi

  "${run_args[@]}"
done

uv run python "${ROOT_DIR}/scripts/plot_wands_llm_as_judge_agreement.py" \
  --input "${RESULTS_CSV}" \
  --output "${PLOT_OUTPUT}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${PLOT_OUTPUT}"

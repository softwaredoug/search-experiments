#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/ecom_judge_variants.csv"
TOOL_CALLS_CSV="${ROOT_DIR}/research/results/ecom_judge_variants_tool_calls.csv"
PLOT_OUTPUT="${ROOT_DIR}/assets/ecom_judge_variants.png"
PARETO_PLOT_OUTPUT="${ROOT_DIR}/assets/ecom_judge_variants_pareto.png"
WORKERS="${WORKERS:-32}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
# Set NO_CACHE=false to reuse strategy results; tool-call collection then falls
# back to the most recent saved trace for each query.
NO_CACHE="${NO_CACHE:-false}"
ESCI_NUM_QUERIES="${ESCI_NUM_QUERIES:-1000}"

CONFIGS=(
  "configs/ecom_base/bm25.yml"
  "configs/ecom_base/embedding_e5_base_v2.yml"
  "configs/ecom_base/agentic_ecom_2tools_e5_gpt5_mini.yml"
  "configs/ecom_judge/agentic_ecom_fewshot_judge_jev_latest.yml"
  "configs/ecom_judge/agentic_ecom_fewshot_judge_gpt5_mini.yml"
)

DATASETS=("esci" "wands")

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT}")"
rm -f "${RESULTS_CSV}"
rm -f "${TOOL_CALLS_CSV}"

cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    strategy="${ROOT_DIR}/${config}"
    echo "Running ${config} on ${dataset}"

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
    fi

    "${run_args[@]}"
  done
done

uv run python "${ROOT_DIR}/scripts/plot_ecom_judge_variants.py" \
  --input "${RESULTS_CSV}" \
  --output "${PLOT_OUTPUT}" \
  --tool-calls-output "${TOOL_CALLS_CSV}" \
  --pareto-output "${PARETO_PLOT_OUTPUT}"

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${TOOL_CALLS_CSV}"
echo "Wrote ${PLOT_OUTPUT}"
echo "Wrote ${PARETO_PLOT_OUTPUT}"

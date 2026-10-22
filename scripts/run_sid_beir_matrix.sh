#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/beir_matrix.csv}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/beir_matrix}"
WORKERS="${WORKERS:-8}"
MODEL_WORKERS="${MODEL_WORKERS:-24}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
# Set NO_CACHE=false to reuse strategy results; the summary CSV is still rebuilt.
NO_CACHE="${NO_CACHE:-false}"
SCIFACT_NUM_QUERIES="${SCIFACT_NUM_QUERIES:-}"
TREC_COVID_NUM_QUERIES="${TREC_COVID_NUM_QUERIES:-}"

BASELINE_CONFIGS=(
  "configs/ecom_base/bm25.yml"
)
SCIFACT_CONFIGS=(
  "configs/ecom_base/bm25.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_gpt6_luna_bm25.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_gpt6_luna_bm25_e5.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_sid_bm25_low.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_sid_bm25_medium.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_sid_bm25_high.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_sid_bm25_e5_low.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_sid_bm25_e5_medium.yml"
  "configs/agentic/scifact_agentic/agentic_scifact_sid_bm25_e5_high.yml"
)
TREC_COVID_CONFIGS=(
  "${BASELINE_CONFIGS[@]}"
  "configs/agentic/beir_sid/agentic_beir_bm25_gpt6_luna.yml"
  "configs/agentic/beir_sid/agentic_beir_bm25_e5_gpt6_luna.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_gpt6_luna_bm25.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_gpt6_luna_bm25_e5.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_sid_bm25_low.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_sid_bm25_medium.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_sid_bm25_high.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_sid_bm25_e5_low.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_sid_bm25_e5_medium.yml"
  "configs/agentic/trec_covid_agentic/agentic_trec_covid_sid_bm25_e5_high.yml"
)
DATASETS=("scifact" "trec_covid")

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT_PREFIX}")"
: > "${RESULTS_CSV}"
cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  if [[ "${dataset}" == "scifact" ]]; then
    dataset_configs=("${SCIFACT_CONFIGS[@]}")
  else
    dataset_configs=("${TREC_COVID_CONFIGS[@]}")
  fi
  for config in "${dataset_configs[@]}"; do
    strategy="${ROOT_DIR}/${config}"
    workers="${WORKERS}"
    if [[ "${config}" == *gpt6* ]]; then
      workers="${MODEL_WORKERS}"
    fi

    echo "Running ${config} on ${dataset} with ${workers} worker(s)"
    run_args=(
      uv run run
      --strategy "${strategy}"
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
    if [[ "${dataset}" == "scifact" && -n "${SCIFACT_NUM_QUERIES}" ]]; then
      run_args+=(--num-queries "${SCIFACT_NUM_QUERIES}")
    elif [[ "${dataset}" == "trec_covid" && -n "${TREC_COVID_NUM_QUERIES}" ]]; then
      run_args+=(--num-queries "${TREC_COVID_NUM_QUERIES}")
    fi

    "${run_args[@]}"
  done
done

uv run python "${ROOT_DIR}/scripts/plot_beir_matrix.py" \
  --input "${RESULTS_CSV}" \
  --output-prefix "${PLOT_OUTPUT_PREFIX}"

echo "Wrote ${RESULTS_CSV}"
for dataset in "${DATASETS[@]}"; do
  echo "Wrote ${PLOT_OUTPUT_PREFIX}_${dataset}.png"
  echo "Wrote ${PLOT_OUTPUT_PREFIX}_pareto_${dataset}.png"
  echo "Wrote ${PLOT_OUTPUT_PREFIX}_cost_pareto_${dataset}.png"
done

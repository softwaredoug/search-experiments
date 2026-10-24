#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/gpt6_luna_wands_bm25_e5_study.csv}"
PER_QUERY_CSV_DIR="${PER_QUERY_CSV_DIR:-${ROOT_DIR}/research/results/gpt6_luna_wands_bm25_e5_study_per_query}"
FULL_RESULTS_PARQUET_DIR="${FULL_RESULTS_PARQUET_DIR:-${TMPDIR:-/tmp}/search-experiments/gpt6_luna_wands_bm25_e5_study_full_results}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/gpt6_luna_wands_bm25_e5_study}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
WORKERS="${WORKERS:-8}"
MODEL_WORKERS="${MODEL_WORKERS:-24}"
DEVICE="${DEVICE:-mps}"
# Bypass result-cache reads so each seed makes fresh searches; results remain cached per seed.
NO_CACHE="${NO_CACHE:-false}"
read -r -a SEEDS <<< "${RUN_SEEDS:-43 44 45 46 47 48 49 50 51}"
if [[ "${#SEEDS[@]}" -ne 9 ]]; then
  echo "RUN_SEEDS must contain exactly nine distinct seeds." >&2
  exit 2
fi
for ((i = 0; i < ${#SEEDS[@]}; i++)); do
  for ((j = i + 1; j < ${#SEEDS[@]}; j++)); do
    if [[ "${SEEDS[i]}" == "${SEEDS[j]}" ]]; then
      echo "RUN_SEEDS must contain exactly nine distinct seeds." >&2
      exit 2
    fi
  done
done

CONFIGS=(
  "configs/agentic/gpt-6-luna-study/agentic_gpt6_luna_bm25_e5.yml"
)
DATASETS=("wands")

mkdir -p \
  "$(dirname "${RESULTS_CSV}")" \
  "$(dirname "${PLOT_OUTPUT_PREFIX}")" \
  "${PER_QUERY_CSV_DIR}" \
  "${FULL_RESULTS_PARQUET_DIR}"
: > "${RESULTS_CSV}"
cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    workers="${WORKERS}"
    if [[ "${config}" == *gpt* ]]; then
      workers="${MODEL_WORKERS}"
    fi

    for seed in "${SEEDS[@]}"; do
      echo "Running ${config} on ${dataset} with seed ${seed} and ${workers} worker(s)"
      strategy_name="${config##*/}"
      strategy_name="${strategy_name%.yml}"
      per_query_csv="${PER_QUERY_CSV_DIR}/${RUN_ID}_${dataset}_${strategy_name}_seed_${seed}.csv"
      full_results_parquet="${FULL_RESULTS_PARQUET_DIR}/${RUN_ID}_${dataset}_${strategy_name}_seed_${seed}.parquet"
      run_args=(
        uv run run
        --strategy "${ROOT_DIR}/${config}"
        --dataset "${dataset}"
        --workers "${workers}"
        --seed "${seed}"
        --summary-csv "${RESULTS_CSV}"
        --per-query-csv "${per_query_csv}"
        --full-results-parquet "${full_results_parquet}"
      )
      if [[ "${NO_CACHE}" == "true" ]]; then
        run_args+=(--no-cache)
      fi
      if [[ -n "${DEVICE}" ]]; then
        run_args+=(--device "${DEVICE}")
      fi
      "${run_args[@]}"
    done
  done
done

uv run python "${ROOT_DIR}/scripts/plot_gpt6_luna_ecom_study.py" \
  --input "${RESULTS_CSV}" \
  --output-prefix "${PLOT_OUTPUT_PREFIX}" \
  --dataset "wands"

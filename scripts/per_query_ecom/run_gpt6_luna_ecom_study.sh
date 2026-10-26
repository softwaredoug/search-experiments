#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/gpt6_luna_wands_bm25_e5_study.csv}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/gpt6_luna_wands_bm25_e5_study}"
WORKERS="${WORKERS:-8}"
MODEL_WORKERS="${MODEL_WORKERS:-24}"
DEVICE="${DEVICE:-mps}"
# Keep query selection fixed and give each stochastic replay its own result cache.
SEED="${SEED:-42}"
NO_CACHE="${NO_CACHE:-false}"
read -r -a CACHE_SALTS <<< "${CACHE_SALTS:-round-1 round-2 round-3 round-4 round-5 round-6 round-7 round-8 round-9}"
if [[ "${#CACHE_SALTS[@]}" -ne 9 ]]; then
  echo "CACHE_SALTS must contain exactly nine distinct values." >&2
  exit 2
fi
for ((i = 0; i < ${#CACHE_SALTS[@]}; i++)); do
  for ((j = i + 1; j < ${#CACHE_SALTS[@]}; j++)); do
    if [[ "${CACHE_SALTS[i]}" == "${CACHE_SALTS[j]}" ]]; then
      echo "CACHE_SALTS must contain exactly nine distinct values." >&2
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
  "$(dirname "${PLOT_OUTPUT_PREFIX}")"
: > "${RESULTS_CSV}"
cd "${ROOT_DIR}"

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    workers="${WORKERS}"
    if [[ "${config}" == *gpt* ]]; then
      workers="${MODEL_WORKERS}"
    fi

    for cache_salt in "${CACHE_SALTS[@]}"; do
      echo "Running ${config} on ${dataset} with seed ${SEED}, cache salt ${cache_salt}, and ${workers} worker(s)"
      run_args=(
        uv run run
        --strategy "${ROOT_DIR}/${config}"
        --dataset "${dataset}"
        --workers "${workers}"
        --seed "${SEED}"
        --cache-salt "${cache_salt}"
        --summary-csv "${RESULTS_CSV}"
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

uv run python "${ROOT_DIR}/scripts/per_query_ecom/plot_gpt6_luna_ecom_study.py" \
  --input "${RESULTS_CSV}" \
  --output-prefix "${PLOT_OUTPUT_PREFIX}" \
  --dataset "wands"

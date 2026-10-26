#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/gpt6_luna_esci_bm25_e5_study.csv}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/gpt6_luna_esci_bm25_e5_study}"
CONFIG="${ROOT_DIR}/configs/agentic/gpt-6-luna-study/agentic_gpt6_luna_bm25_e5.yml"
NUM_QUERIES="${NUM_QUERIES:-400}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-24}"
DEVICE="${DEVICE:-mps}"
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

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT_PREFIX}")"
: > "${RESULTS_CSV}"
cd "${ROOT_DIR}"

for cache_salt in "${CACHE_SALTS[@]}"; do
  echo "Running GPT-6 Luna BM25+E5 on ESCI (${NUM_QUERIES} queries), seed ${SEED}, cache salt ${cache_salt}"
  run_args=(
    uv run run
    --strategy "${CONFIG}"
    --dataset esci
    --num-queries "${NUM_QUERIES}"
    --seed "${SEED}"
    --cache-salt "${cache_salt}"
    --workers "${WORKERS}"
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

uv run python "${ROOT_DIR}/scripts/per_query_ecom/plot_gpt6_luna_ecom_study.py" \
  --input "${RESULTS_CSV}" \
  --output-prefix "${PLOT_OUTPUT_PREFIX}" \
  --dataset esci

uv run python "${ROOT_DIR}/scripts/per_query_ecom/analyze_gpt6_luna_esci_study.py" \
  --input "${RESULTS_CSV}" \
  --by-quartile 0.25 \
  --by-quartile-num-rounds 9

echo "Wrote ${RESULTS_CSV}"

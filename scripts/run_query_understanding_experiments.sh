#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/results_query_understanding.csv}"
CLASSIFICATION_CSV="${CLASSIFICATION_CSV:-${ROOT_DIR}/query_understanding_classification_wands.csv}"
SEED="${SEED:-42}"
WORKERS="${WORKERS:-4}"
NO_CACHE="${NO_CACHE:-false}"

CONFIGS=(
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_bm25_filtered_llm_single.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_bm25_boosted_llm_single.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_bm25_hierarchy_boosted_llm_single.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_hierarchy_bm25_filtered_llm_single.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_hierarchy_bm25_boosted_llm_single.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_hierarchy_bm25_hierarchy_boosted_llm_single.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_bm25_filtered_llm_multiple.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_bm25_boosted_llm_multiple.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_bm25_hierarchy_boosted_llm_multiple.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_hierarchy_bm25_filtered_llm_multiple.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_hierarchy_bm25_boosted_llm_multiple.yml"
  "${ROOT_DIR}/configs/ecom_class/ecom_query_understanding_category_hierarchy_bm25_hierarchy_boosted_llm_multiple.yml"
)

mkdir -p "$(dirname "${RESULTS_CSV}")"
: > "${RESULTS_CSV}"
mkdir -p "$(dirname "${CLASSIFICATION_CSV}")"
: > "${CLASSIFICATION_CSV}"

for config in "${CONFIGS[@]}"; do
  args=(
    --strategy "${config}"
    --dataset wands
    --seed "${SEED}"
    --workers "${WORKERS}"
    --summary-csv "${RESULTS_CSV}"
  )
  if [[ "${NO_CACHE}" == "true" ]]; then
    args+=(--no-cache)
  fi

  echo "Running $(basename "${config}") against wands"
  uv run run "${args[@]}"

  echo "Collecting classification stats for $(basename "${config}")"
  uv run query_classification \
    --strategy "${config}" \
    --dataset wands \
    --eval-as "taxonomy[0],taxonomy[1],direct" \
    --query-threshold 0.4 \
    --workers "${WORKERS}" \
    --summary-csv "${CLASSIFICATION_CSV}"
done

echo "Wrote ${RESULTS_CSV}"
echo "Wrote ${CLASSIFICATION_CSV}"

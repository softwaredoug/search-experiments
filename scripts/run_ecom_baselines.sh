#!/usr/bin/env bash
set -euo pipefail

RESULTS_CSV="results.csv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXIT_LOG_WRAPPER="${SCRIPT_DIR}/run_with_exit_log.sh"

run_baseline() {
  local dataset="$1"
  local strategy="$2"
  shift 2
  echo "Running: uv run run --strategy \"${strategy}\" --dataset \"${dataset}\" --summary-csv \"${RESULTS_CSV}\" $*"
  "$EXIT_LOG_WRAPPER" --time uv run run --strategy "$strategy" --dataset "$dataset" --summary-csv "$RESULTS_CSV" "$@"
}

run_baseline "esci" "configs/ecom_base/bm25.yml" --num-queries 1000
run_baseline "esci" "configs/ecom_base/embedding_minilm.yml" --num-queries 1000
run_baseline "esci" "configs/ecom_base/embedding_e5_base_v2.yml" --num-queries 1000

run_baseline "wands" "configs/ecom_base/bm25.yml"
run_baseline "wands" "configs/ecom_base/embedding_minilm.yml"
run_baseline "wands" "configs/ecom_base/embedding_e5_base_v2.yml"

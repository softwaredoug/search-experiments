#!/usr/bin/env bash
set -euo pipefail

RESULTS_CSV="results.csv"

run_baseline() {
  local dataset="$1"
  local strategy="$2"
  shift 2
  echo "Running: uv run run --strategy \"${strategy}\" --dataset \"${dataset}\" --summary-csv \"${RESULTS_CSV}\" $*"
  /usr/bin/time -l uv run run --strategy "$strategy" --dataset "$dataset" --summary-csv "$RESULTS_CSV" "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Command failed with exit_code=${status}"
    if [ $status -gt 128 ]; then
      echo "signal=$((status-128))"
    fi
  fi
}

run_baseline "esci" "configs/ecom_base/bm25.yml" --num-queries 1000
run_baseline "esci" "configs/ecom_base/embedding_minilm.yml" --num-queries 1000
run_baseline "esci" "configs/ecom_base/embedding_e5_base_v2.yml" --num-queries 1000

run_baseline "wands" "configs/ecom_base/bm25.yml"
run_baseline "wands" "configs/ecom_base/embedding_minilm.yml"
run_baseline "wands" "configs/ecom_base/embedding_e5_base_v2.yml"

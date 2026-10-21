#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/beir_matrix.csv}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/beir_matrix}"
WORKERS="${WORKERS:-8}"
MODEL_WORKERS="${MODEL_WORKERS:-24}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
NO_CACHE="${NO_CACHE:-false}"
RERUN_BEIR_MATRIX="${RERUN_BEIR_MATRIX:-false}"
SCIFACT_NUM_QUERIES="${SCIFACT_NUM_QUERIES:-}"
TREC_COVID_NUM_QUERIES="${TREC_COVID_NUM_QUERIES:-}"

BASELINE_CONFIGS=(
  "configs/ecom_base/bm25.yml"
)
GPT5_MINI_CONFIGS=(
  "configs/beir_sid/agentic_beir_bm25_gpt5_mini.yml"
  "configs/beir_sid/agentic_beir_bm25_e5_gpt5_mini.yml"
)
GPT6_LUNA_CONFIGS=(
  "configs/beir_sid/agentic_beir_bm25_gpt6_luna.yml"
  "configs/beir_sid/agentic_beir_bm25_e5_gpt6_luna.yml"
)
SID_CONFIGS=(
  "configs/beir_sid/agentic_beir_sid_bm25_low.yml"
  "configs/beir_sid/agentic_beir_sid_bm25_medium.yml"
  "configs/beir_sid/agentic_beir_sid_bm25_high.yml"
  "configs/beir_sid/agentic_beir_sid_bm25_e5_low.yml"
  "configs/beir_sid/agentic_beir_sid_bm25_e5_medium.yml"
  "configs/beir_sid/agentic_beir_sid_bm25_e5_high.yml"
)
CONFIGS=(
  "${BASELINE_CONFIGS[@]}"
  "${GPT5_MINI_CONFIGS[@]}"
  "${GPT6_LUNA_CONFIGS[@]}"
  "${SID_CONFIGS[@]}"
)
DATASETS=("scifact" "trec_covid")
STRATEGY_NAMES=(
  "bm25"
  "agentic_bm25_ecommerce_gpt5_mini"
  "agentic_bm25_e5_ecommerce_gpt5_mini"
  "agentic_bm25_ecommerce_gpt6_luna"
  "agentic_bm25_e5_ecommerce_gpt6_luna"
  "agentic_ecom_sid_bm25_low"
  "agentic_ecom_sid_bm25_medium"
  "agentic_ecom_sid_bm25_high"
  "agentic_ecom_sid_bm25_e5_base_v2_low"
  "agentic_ecom_sid_bm25_e5_base_v2_medium"
  "agentic_ecom_sid_bm25_e5_base_v2_high"
)

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT_PREFIX}")"
cd "${ROOT_DIR}"

filter_existing_results() {
  uv run python - "${RESULTS_CSV}" "${DATASETS[@]}" -- "${STRATEGY_NAMES[@]}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
separator = sys.argv.index("--")
datasets = set(sys.argv[2:separator])
strategies = set(sys.argv[separator + 1 :])
if not path.exists():
    raise SystemExit(0)

with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames or not {"dataset", "strategy_name"}.issubset(
        reader.fieldnames
    ):
        raise SystemExit(f"Invalid BEIR matrix results CSV: {path}")
    fieldnames = reader.fieldnames
    rows = [
        row
        for row in reader
        if not (row["dataset"] in datasets and row["strategy_name"] in strategies)
    ]

with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

filter_existing_results

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    strategy="${ROOT_DIR}/${config}"
    workers="${WORKERS}"
    if [[ "${config}" == *gpt5* || "${config}" == *gpt6* ]]; then
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
    if [[ "${NO_CACHE}" == "true" || "${RERUN_BEIR_MATRIX}" == "true" ]]; then
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

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${RESULTS_CSV:-${ROOT_DIR}/research/results/ecom_sid_matrix.csv}"
PLOT_OUTPUT_PREFIX="${PLOT_OUTPUT_PREFIX:-${ROOT_DIR}/assets/ecom_sid_matrix}"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
NO_CACHE="${NO_CACHE:-false}"
# Set true to rerun GPT-5-mini and SID variants and bypass their result cache.
RERUN_SID_VARIANTS="${RERUN_SID_VARIANTS:-false}"
ESCI_NUM_QUERIES="${ESCI_NUM_QUERIES:-1000}"
WANDS_NUM_QUERIES="${WANDS_NUM_QUERIES:-}"

BASELINE_CONFIGS=(
  "configs/ecom_base/bm25.yml"
)
GPT5_MINI_CONFIGS=(
  "configs/ecom_agentic/agentic_ecom_bm25_gpt5_mini.yml"
  "configs/ecom_agentic/agentic_ecom_2tools_e5_gpt5_mini.yml"
)
SID_CONFIGS=(
  "configs/ecom_sid/agentic_ecom_sid_bm25_low.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_medium.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_high.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_low.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_medium.yml"
  "configs/ecom_sid/agentic_ecom_sid_bm25_e5_base_v2_high.yml"
)
CONFIGS=("${BASELINE_CONFIGS[@]}" "${GPT5_MINI_CONFIGS[@]}" "${SID_CONFIGS[@]}")
DATASETS=("wands" "esci")

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT_PREFIX}")"
cd "${ROOT_DIR}"

filter_existing_results() {
  local mode="$1"
  uv run python - "${RESULTS_CSV}" "${mode}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[2] if len(sys.argv) > 2 else ""
full_gpt5 = {
    "agentic_bm25_ecommerce",
    "agentic_bm25_e5_ecommerce_gpt5",
}
refresh_strategies = {
    "agentic_bm25_ecommerce_gpt5_mini",
    "agentic_bm25_e5_ecommerce_gpt5_mini",
    "agentic_ecom_sid_bm25_low",
    "agentic_ecom_sid_bm25_medium",
    "agentic_ecom_sid_bm25_high",
    "agentic_ecom_sid_bm25_e5_base_v2_low",
    "agentic_ecom_sid_bm25_e5_base_v2_medium",
    "agentic_ecom_sid_bm25_e5_base_v2_high",
}
if not path.exists():
    raise SystemExit(0)

with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames or "strategy_name" not in reader.fieldnames:
        raise SystemExit(f"Invalid matrix results CSV: {path}")
    fieldnames = reader.fieldnames
    if mode == "rerun":
        rows = [row for row in reader if row["strategy_name"] not in refresh_strategies]
    else:
        rows = [row for row in reader if row["strategy_name"] in full_gpt5]

with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
}

if [[ "${RERUN_SID_VARIANTS}" == "true" ]]; then
  filter_existing_results rerun
  CONFIGS=("${GPT5_MINI_CONFIGS[@]}" "${SID_CONFIGS[@]}")
else
  filter_existing_results full
fi

for dataset in "${DATASETS[@]}"; do
  for config in "${CONFIGS[@]}"; do
    strategy="${ROOT_DIR}/${config}"

    echo "Running ${config} on ${dataset} with ${WORKERS} worker(s)"
    run_args=(
      uv run run
      --strategy "${strategy}"
      --dataset "${dataset}"
      --workers "${WORKERS}"
      --seed "${SEED}"
      --summary-csv "${RESULTS_CSV}"
    )
    if [[ "${NO_CACHE}" == "true" || "${RERUN_SID_VARIANTS}" == "true" ]]; then
      run_args+=(--no-cache)
    fi
    if [[ -n "${DEVICE}" ]]; then
      run_args+=(--device "${DEVICE}")
    fi
    if [[ "${dataset}" == "esci" ]]; then
      run_args+=(--num-queries "${ESCI_NUM_QUERIES}")
    elif [[ "${dataset}" == "wands" && -n "${WANDS_NUM_QUERIES}" ]]; then
      run_args+=(--num-queries "${WANDS_NUM_QUERIES}")
    fi

    "${run_args[@]}"
  done
done

uv run python "${ROOT_DIR}/scripts/plot_ecom_sid_matrix.py" \
  --input "${RESULTS_CSV}" \
  --output-prefix "${PLOT_OUTPUT_PREFIX}"

echo "Wrote ${RESULTS_CSV}"
for dataset in "${DATASETS[@]}"; do
  echo "Wrote ${PLOT_OUTPUT_PREFIX}_${dataset}.png"
  echo "Wrote ${PLOT_OUTPUT_PREFIX}_pareto_${dataset}.png"
done

#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/ecom_sid_matrix.csv"
PLOT_OUTPUT_PREFIX="${ROOT_DIR}/assets/ecom_sid_matrix"
WORKERS="${WORKERS:-8}"
DEVICE="${DEVICE:-mps}"
SEED="${SEED:-42}"
NO_CACHE="${NO_CACHE:-false}"
# Set true to rerun only SID variants and bypass their result cache.
RERUN_SID_VARIANTS="${RERUN_SID_VARIANTS:-false}"
ESCI_NUM_QUERIES="${ESCI_NUM_QUERIES:-1000}"
WANDS_NUM_QUERIES="${WANDS_NUM_QUERIES:-}"

CONFIGS=(
  "configs/ecom_base/bm25.yml"
  "configs/agentic_ecom_bm25_gpt5.yml"
  "configs/ecom_agentic/agentic_ecom_2tools_gpt5.yml"
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
CONFIGS+=("${SID_CONFIGS[@]}")
DATASETS=("wands" "esci")

mkdir -p "$(dirname "${RESULTS_CSV}")" "$(dirname "${PLOT_OUTPUT_PREFIX}")"
cd "${ROOT_DIR}"

if [[ "${RERUN_SID_VARIANTS}" == "true" ]]; then
  if [[ ! -f "${RESULTS_CSV}" ]]; then
    echo "Cannot rerun SID variants without existing matrix results: ${RESULTS_CSV}" >&2
    exit 1
  fi
  uv run python - "${RESULTS_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
with path.open(encoding="utf-8", newline="") as handle:
    reader = csv.DictReader(handle)
    if not reader.fieldnames or "strategy_name" not in reader.fieldnames:
        raise SystemExit(f"Invalid matrix results CSV: {path}")
    fieldnames = reader.fieldnames
    rows = [
        row
        for row in reader
        if not row["strategy_name"].startswith("agentic_ecom_sid_")
    ]

if not rows:
    raise SystemExit(
        "No non-SID matrix rows found; run the full matrix once before "
        "setting RERUN_SID_VARIANTS=true."
    )

with path.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
PY
  CONFIGS=("${SID_CONFIGS[@]}")
else
  rm -f "${RESULTS_CSV}"
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

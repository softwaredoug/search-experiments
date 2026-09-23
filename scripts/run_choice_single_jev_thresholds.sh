#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${BASE_CONFIG:-${ROOT_DIR}/configs/ecom_class/ecom_choice_single.yml}"
GPT5_CONFIG="${GPT5_CONFIG:-${ROOT_DIR}/configs/ecom_class/ecom_choice_single_gpt5.yml}"
DATASET="${DATASET:-wands}"
WORKERS="${WORKERS:-1}"
QUERY_THRESHOLD="${QUERY_THRESHOLD:-0.8}"
OUTPUT_CSV="${OUTPUT_CSV:-${ROOT_DIR}/research/results/query_understanding_jev_thresholds.csv}"
PLOT_OUTPUT="${PLOT_OUTPUT:-${ROOT_DIR}/assets/query_understanding_jev_thresholds.png}"
RAW_CSV="$(mktemp "${TMPDIR:-/tmp}/choice-single-jev.XXXXXX.csv")"
CONFIG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/choice-single-jev-configs.XXXXXX")"

cleanup() {
  rm -f "${RAW_CSV}"
  rm -rf "${CONFIG_DIR}"
}
trap cleanup EXIT

mkdir -p "$(dirname "${OUTPUT_CSV}")" "$(dirname "${PLOT_OUTPUT}")"

run_classification() {
  local config="$1"
  echo "Running $(basename "${config}") against ${DATASET}"
  uv run query_classification \
    --strategy "${config}" \
    --dataset "${DATASET}" \
    --eval-as direct \
    --query-threshold "${QUERY_THRESHOLD}" \
    --workers "${WORKERS}" \
    --summary-csv "${RAW_CSV}"
}

run_classification "${BASE_CONFIG}"
run_classification "${GPT5_CONFIG}"

for confidence_threshold in 0.6 0.7 0.8 0.9 0.95 0.99; do
  threshold_label="${confidence_threshold/./_}"
  variant_config="${CONFIG_DIR}/ecom_choice_single_jev_${threshold_label}.yml"
  sed \
    -e '/^[[:space:]]*threshold:/d' \
    -e "s/name: ecom_choice_single$/name: ecom_choice_single_jev_${threshold_label}/" \
    -e 's/model: gpt-5-mini/model: jev\/jev-latest/' \
    -e "/model: jev\/jev-latest/a\\
          confidence_threshold: ${confidence_threshold}" \
    "${BASE_CONFIG}" > "${variant_config}"
  run_classification "${variant_config}"
done

uv run python - "${RAW_CSV}" "${OUTPUT_CSV}" <<'PY'
import csv
import re
import sys

raw_path, output_path = sys.argv[1:]
rows = []
with open(raw_path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("eval_as") != "direct":
            continue
        strategy = row["strategy"]
        match = re.search(r"_jev_(\d+_\d+)$", strategy)
        if strategy == "ecom_choice_single_gpt5":
            threshold = "gpt-5"
        else:
            threshold = match.group(1).replace("_", ".") if match else "baseline"
        rows.append(
            {
                "variant": strategy,
                "confidence_threshold": threshold,
                "recall": row["mean_recall"],
                "coverage": row["coverage"],
            }
        )

with open(output_path, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=["variant", "confidence_threshold", "recall", "coverage"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
PY

uv run python "${ROOT_DIR}/scripts/plot_choice_single_jev_thresholds.py" \
  --input "${OUTPUT_CSV}" \
  --output "${PLOT_OUTPUT}"

echo "Wrote ${OUTPUT_CSV}"
echo "Wrote ${PLOT_OUTPUT}"

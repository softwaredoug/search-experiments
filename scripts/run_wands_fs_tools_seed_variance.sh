#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/wands_fs_tools_seed_variance.csv"

mkdir -p "$(dirname "${RESULTS_CSV}")"
rm -f "${RESULTS_CSV}"

for seed in {42..49}; do
  uv run run \
    --strategy "${ROOT_DIR}/configs/wands_grep/agentic_wands_fs_tools.yml" \
    --dataset "wands" \
    --seed "${seed}" \
    --workers 16 \
    --summary-csv "${RESULTS_CSV}"
done

python - "${RESULTS_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit("No results CSV found.")

with path.open(newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

if not rows:
    raise SystemExit("No rows in results CSV.")

def pick_ndcg(row: dict) -> tuple[str | None, str | None]:
    for key in row:
        if "ndcg" in key.lower():
            return key, row.get(key)
    return None, None

print("seed\tndcg")
for row in rows:
    seed = row.get("seed") or ""
    key, value = pick_ndcg(row)
    if value is None:
        value = ""
    print(f"{seed}\t{value}")
PY

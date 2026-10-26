#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_CSV="${ROOT_DIR}/research/results/wands_fs_tools_seed_variance.csv"
SEED="${SEED:-42}"
read -r -a CACHE_SALTS <<< "${CACHE_SALTS:-round-1 round-2 round-3 round-4 round-5 round-6 round-7 round-8}"

mkdir -p "$(dirname "${RESULTS_CSV}")"
rm -f "${RESULTS_CSV}"

for cache_salt in "${CACHE_SALTS[@]}"; do
  uv run run \
    --strategy "${ROOT_DIR}/configs/agentic/wands_grep/agentic_wands_fs_tools.yml" \
    --dataset "wands" \
    --seed "${SEED}" \
    --cache-salt "${cache_salt}" \
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

def pick_ndcg_m(row: dict) -> tuple[str | None, str | None]:
    for key in ("mean_ndcg_m", "mean_ndcg"):
        if key in row:
            return key, row.get(key)
    return None, None

print("cache_salt\tndcg_m")
for row in rows:
    cache_salt = row.get("cache_salt") or ""
    key, value = pick_ndcg_m(row)
    if value is None:
        value = ""
    print(f"{cache_salt}\t{value}")
PY

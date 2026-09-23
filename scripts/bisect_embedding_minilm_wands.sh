#!/usr/bin/env bash
set -euo pipefail

CACHE_DIR="${HOME}/.search-experiments/embeddings"
rm -rf "${CACHE_DIR}"

if ! uv sync --frozen; then
  exit 125
fi

export OUTPUT
OUTPUT=$(uv run run --strategy configs/ecom_base/embedding_minilm.yml --dataset wands)
echo "${OUTPUT}"

python - <<'PY'
import os
import re
import sys

output = os.environ.get("OUTPUT", "")
match = re.search(r"mean_ndcg=([0-9]*\.[0-9]+)", output)
if not match:
    sys.exit(125)

mean = float(match.group(1))
target = float(os.environ.get("TARGET_NDCG", "0.5060"))
tol = float(os.environ.get("NDCG_TOLERANCE", "0.01"))

if mean >= target - tol:
    sys.exit(0)
sys.exit(1)
PY

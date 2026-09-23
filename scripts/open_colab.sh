#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ] || [ $# -gt 2 ]; then
  echo "Usage: open_colab.sh <notebook-path> [github-repo-url]" >&2
  exit 1
fi

NOTEBOOK_PATH="$1"
GITHUB_URL="${2:-https://github.com/softwaredoug/search-experiments}"
COLAB_BRANCH="${COLAB_BRANCH:-main}"

REPO_ROOT=$(git rev-parse --show-toplevel)

RELATIVE_PATH=$(python - "$REPO_ROOT" "$NOTEBOOK_PATH" <<'PY'
import os
import sys

repo_root = sys.argv[1]
notebook_path = sys.argv[2]

abs_path = os.path.abspath(notebook_path)
rel_path = os.path.relpath(abs_path, repo_root)

if rel_path.startswith(".."):
    raise SystemExit("Notebook path must be inside the repo.")

print(rel_path)
PY
)

REPO_PATH=$(python - "$GITHUB_URL" <<'PY'
import re
import sys
from urllib.parse import urlparse

github_url = sys.argv[1]
parsed = urlparse(github_url)
path = parsed.path.strip("/")

if not path:
    raise SystemExit("Invalid GitHub URL.")

path = re.sub(r"\.git$", "", path)
print(path)
PY
)

COLAB_URL="https://colab.research.google.com/github/${REPO_PATH}/blob/${COLAB_BRANCH}/${RELATIVE_PATH}"

echo "$COLAB_URL"
if command -v open >/dev/null 2>&1; then
  open "$COLAB_URL" >/dev/null 2>&1 || true
fi

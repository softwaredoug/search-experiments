#!/usr/bin/env bash
set -euo pipefail

strategy_config="configs/bm25_strong_title.yml"
strategy_name=""
dataset="wands"
query="salon chair"
k="10"
bm25_k1=""
bm25_b=""

usage() {
  cat <<'EOF'
Usage: scripts/test_bm25_consistency.sh [options]

Options:
  --strategy-config P  Strategy config path (default: configs/bm25_strong_title.yml)
  --strategy-name N    Strategy name override (defaults to config file stem)
  --dataset NAME       Dataset (default: wands)
  --query TEXT         Query to test (default: "salon chair")
  --k N                Number of results (default: 10)
  --bm25-k1 N           BM25 k1 parameter (required)
  --bm25-b N            BM25 b parameter (required)
  -h, --help           Show this help message
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy-config)
      strategy_config="$2"
      shift 2
      ;;
    --strategy-config=*)
      strategy_config="${1#*=}"
      shift 1
      ;;
    --strategy-name)
      strategy_name="$2"
      shift 2
      ;;
    --strategy-name=*)
      strategy_name="${1#*=}"
      shift 1
      ;;
    --dataset)
      dataset="$2"
      shift 2
      ;;
    --dataset=*)
      dataset="${1#*=}"
      shift 1
      ;;
    --query)
      query="$2"
      shift 2
      ;;
    --query=*)
      query="${1#*=}"
      shift 1
      ;;
    --k)
      k="$2"
      shift 2
      ;;
    --k=*)
      k="${1#*=}"
      shift 1
      ;;
    --bm25-k1)
      bm25_k1="$2"
      shift 2
      ;;
    --bm25-k1=*)
      bm25_k1="${1#*=}"
      shift 1
      ;;
    --bm25-b)
      bm25_b="$2"
      shift 2
      ;;
    --bm25-b=*)
      bm25_b="${1#*=}"
      shift 1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$bm25_k1" || -z "$bm25_b" ]]; then
  echo "--bm25-k1 and --bm25-b are required" >&2
  usage
  exit 1
fi

if [[ -z "$strategy_name" ]]; then
  filename=$(basename "$strategy_config")
  strategy_name="${filename%.yml}"
  strategy_name="${strategy_name%.yaml}"
fi

debug_strategy="bm25"
score_field="bm25_score"

tmp_dir=$(mktemp -d)
diff_out="$tmp_dir/diff.txt"
debug_out="$tmp_dir/debug.txt"
query_out="$tmp_dir/query.txt"
trap 'rm -rf "$tmp_dir"' EXIT

uv run diff \
  --strategy-a "$strategy_config" \
  --strategy-b "$strategy_config" \
  --dataset "$dataset" \
  --query "$query" \
  --k "$k" > "$diff_out"

uv run bm25-debug \
  --strategy "$debug_strategy" \
  --dataset "$dataset" \
  --query "$query" \
  --k "$k" \
  --bm25-k1 "$bm25_k1" \
  --bm25-b "$bm25_b" > "$debug_out"

uv run query \
  --strategy "$strategy_config" \
  --dataset "$dataset" \
  --query "$query" \
  --k "$k" > "$query_out"

STRATEGY="$strategy_name" \
SCORE_FIELD="$score_field" \
DIFF_OUT="$diff_out" \
DEBUG_OUT="$debug_out" \
QUERY_OUT="$query_out" \
python - <<'PY'
import os
import sys

strategy = os.environ["STRATEGY"]
score_field = os.environ["SCORE_FIELD"]
diff_path = os.environ["DIFF_OUT"]
debug_path = os.environ["DEBUG_OUT"]
query_path = os.environ["QUERY_OUT"]


def parse_diff(path, label):
    results = []
    in_block = False
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == f"{label} results:":
                in_block = True
                continue
            if not in_block:
                continue
            if "\t" not in line:
                break
            parts = line.split("\t")
            if len(parts) < 2:
                break
            doc_id = parts[0]
            score = float(parts[1])
            results.append((doc_id, score))
    return results


def parse_debug(path, score_column):
    results = []
    header = None
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("doc_id\t"):
                header = line.split("\t")
                continue
            if header is None:
                continue
            if not line:
                break
            if line.startswith("Sum "):
                break
            parts = line.split("\t")
            if len(parts) < len(header):
                break
            row = dict(zip(header, parts))
            if "doc_id" not in row:
                continue
            doc_id = row["doc_id"]
            score = float(row[score_column])
            results.append((doc_id, score))
    return results


def parse_query(path):
    results = []
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line.startswith("Query:"):
                continue
            if "\t" not in line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            doc_id = parts[0]
            score = float(parts[1])
            results.append((doc_id, score))
    return results


def compare(name_a, a, name_b, b, tol=1e-4):
    if len(a) != len(b):
        raise AssertionError(f"Length mismatch {name_a}={len(a)} {name_b}={len(b)}")
    for idx, ((doc_a, score_a), (doc_b, score_b)) in enumerate(zip(a, b)):
        if doc_a != doc_b:
            raise AssertionError(
                f"Doc mismatch at {idx}: {name_a}={doc_a} {name_b}={doc_b}"
            )
        if abs(score_a - score_b) > tol:
            raise AssertionError(
                f"Score mismatch at {idx} ({doc_a}): {name_a}={score_a} {name_b}={score_b}"
            )


diff_results = parse_diff(diff_path, strategy)
debug_results = parse_debug(debug_path, score_field)
query_results = parse_query(query_path)

if not diff_results:
    raise SystemExit(f"No results parsed from diff output: {diff_path}")
if not debug_results:
    raise SystemExit(f"No results parsed from bm25-debug output: {debug_path}")
if not query_results:
    raise SystemExit(f"No results parsed from query output: {query_path}")

compare("diff", diff_results, "bm25-debug", debug_results)
compare("diff", diff_results, "query", query_results)

print(
    "Consistency check passed for",
    strategy,
    "using",
    score_field,
    f"({len(diff_results)} results)",
)
PY

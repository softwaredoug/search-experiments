#!/usr/bin/env bash
set -euo pipefail

strategy_a="bm25"
strategy_b="bm25_reweighed"
dataset="msmarco"
num_queries="5"
out_dir="$HOME/.search-experiments"

usage() {
  cat <<'EOF'
Usage: scripts/profile_memray.sh [options]

Options:
  --strategy-a NAME     Strategy A (default: bm25)
  --strategy-b NAME     Strategy B (default: bm25_reweighed)
  --dataset NAME        Dataset (default: msmarco)
  --num-queries N       Number of queries (default: 5)
  --out-dir PATH        Output directory (default: ~/.search-experiments)
  --leaks               Trace python allocators and show leaks flamegraph
  -h, --help            Show this help message
EOF
}

trace_leaks=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --strategy-a)
      strategy_a="$2"
      shift 2
      ;;
    --strategy-a=*)
      strategy_a="${1#*=}"
      shift 1
      ;;
    --strategy-b)
      strategy_b="$2"
      shift 2
      ;;
    --strategy-b=*)
      strategy_b="${1#*=}"
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
    --num-queries)
      num_queries="$2"
      shift 2
      ;;
    --num-queries=*)
      num_queries="${1#*=}"
      shift 1
      ;;
    --out-dir)
      out_dir="$2"
      shift 2
      ;;
    --out-dir=*)
      out_dir="${1#*=}"
      shift 1
      ;;
    --leaks)
      trace_leaks=true
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

mkdir -p "$out_dir"

timestamp=$(date "+%Y%m%d_%H%M%S")
base="memray_${strategy_a}_vs_${strategy_b}_${dataset}_n${num_queries}_${timestamp}"
bin_path="$out_dir/${base}.bin"
html_path="$out_dir/memray-flamegraph-${base}.html"

trace_flag=""
flamegraph_flag=""
if [[ "$trace_leaks" == "true" ]]; then
  trace_flag="--trace-python-allocators"
  flamegraph_flag="--leaks"
fi

uv run memray run $trace_flag --follow-fork -o "$bin_path" -m prf.diff -- \
  --strategy-a "$strategy_a" \
  --strategy-b "$strategy_b" \
  --dataset "$dataset" \
  --num-queries "$num_queries"

uv run memray summary "$bin_path"
uv run memray flamegraph $flamegraph_flag -o "$html_path" "$bin_path"

open "$html_path"

echo "Memray binary: $bin_path"
echo "Flamegraph: $html_path"

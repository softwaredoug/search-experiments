#!/usr/bin/env bash
set -euo pipefail

URL="http://127.0.0.1:8000/execute"

run_req() {
  local name="$1"
  local cmd="$2"
  local start
  start=$(date +%s)
  local payload
  payload=$(python -c "import json,sys; print(json.dumps({'command': sys.argv[1], 'timeout': 120}))" "$cmd")
  curl -s -X POST "$URL" \
    -H "Content-Type: application/json" \
    -d "$payload" >/dev/null
  local end
  end=$(date +%s)
  echo "$name done in $((end - start))s"
}

while true; do
  run_req "grep1" 'grep -Rli "certified international" . | head -n 200' &
  run_req "grep2" 'grep -Rli "melamine" . | head -n 200' &
  run_req "grep3" 'grep -RIn "international" -n . || true' &
  run_req "grep4" 'grep -RIn "certified-international" -n . || true' &
  wait
done

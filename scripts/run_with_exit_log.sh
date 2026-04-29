#!/usr/bin/env bash
set -euo pipefail

use_time=false
if [[ "${1-}" == "--time" ]]; then
  use_time=true
  shift
fi
if [[ "${1-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  echo "usage: run_with_exit_log.sh [--time] -- <command...>"
  exit 2
fi

start_ts=$(date -u "+%Y-%m-%dT%H:%M:%SZ")
echo "start=${start_ts}"
echo "cmd=$*"

set +e
if $use_time; then
  /usr/bin/time -l "$@"
  status=$?
else
  "$@"
  status=$?
fi
set -e

end_ts=$(date -u "+%Y-%m-%dT%H:%M:%SZ")
echo "end=${end_ts}"
echo "exit_code=${status}"
if [ $status -gt 128 ]; then
  echo "signal=$((status-128))"
fi

exit $status

#!/usr/bin/env bash
set -euo pipefail

workflow_file="${1:-tests.yml}"
job_name="${2:-integration-tests}"
mode="${3:-fail}"

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found in PATH" >&2
  exit 1
fi

run_id=$(gh run list --workflow "$workflow_file" --limit 10 --json databaseId,status --jq 'map(select(.status == "completed")) | .[0].databaseId')

if [ -z "$run_id" ] || [ "$run_id" = "null" ]; then
  echo "No runs found for workflow $workflow_file" >&2
  exit 1
fi

gh run list --workflow "$workflow_file" --limit 10 --json displayTitle,headBranch,headSha,createdAt,status,conclusion \
  --jq 'map(select(.status == "completed")) | .[0] | "Run: \(.displayTitle)\nBranch: \(.headBranch)\nSHA: \(.headSha)\nCreated: \(.createdAt)\nStatus: \(.status)\nConclusion: \(.conclusion)"'

job_status=$(gh run view "$run_id" --json jobs --jq ".jobs[] | select(.name == \"$job_name\") | \"Job: \(.name)\nStatus: \(.status)\nConclusion: \(.conclusion)\nURL: \(.url)\"")

if [ -z "$job_status" ]; then
  echo "Job '$job_name' not found in run $run_id" >&2
  exit 1
fi

printf "%s\n" "$job_status"

job_id=$(gh run view "$run_id" --json jobs --jq ".jobs[] | select(.name == \"$job_name\") | .databaseId")
job_conclusion=$(gh run view "$run_id" --json jobs --jq ".jobs[] | select(.name == \"$job_name\") | .conclusion")

if [ -n "$job_id" ] && [ "$job_conclusion" != "success" ] && [ "$job_conclusion" != "null" ]; then
  printf "\nFailed step logs (%s):\n" "$job_name"
  gh run view "$run_id" --job "$job_id" --log-failed
  if [ "$mode" = "warn" ]; then
    printf "\nWarning: last %s job concluded with %s\n" "$job_name" "$job_conclusion" >&2
    exit 0
  fi
  exit 1
fi

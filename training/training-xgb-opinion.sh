#!/usr/bin/env bash
# Compatibility wrapper for tooling that previously targeted opinion-only runs.
# training-xgb.sh now schedules both next-video and opinion tasks by default.

set -euo pipefail

if [[ -n "${TRAINING_REPO_ROOT:-}" ]]; then
  ROOT_DIR=$(realpath "${TRAINING_REPO_ROOT}")
  SCRIPT_DIR="${ROOT_DIR}/training"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XGB_PIPELINE_TASKS="next_video,opinion"
exec "${SCRIPT_DIR}/training-xgb.sh" "$@"

#!/usr/bin/env bash
# Wrapper that runs the XGBoost pipeline for next-video tasks only.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XGB_PIPELINE_TASKS="next_video"
exec "${SCRIPT_DIR}/training-xgb.sh" "$@"

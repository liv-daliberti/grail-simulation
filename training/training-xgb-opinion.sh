#!/usr/bin/env bash
# Wrapper that runs the XGBoost pipeline for opinion tasks only.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export XGB_PIPELINE_TASKS="opinion"
exec "${SCRIPT_DIR}/training-xgb.sh" "$@"

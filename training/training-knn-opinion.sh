#!/usr/bin/env bash
# Wrapper that runs the KNN pipeline for opinion tasks only.

set -euo pipefail

if [[ -n "${TRAINING_REPO_ROOT:-}" ]]; then
  ROOT_DIR=$(realpath "${TRAINING_REPO_ROOT}")
  SCRIPT_DIR="${ROOT_DIR}/training"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)
fi
export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
export KNN_PIPELINE_TASKS="opinion"
exec "${SCRIPT_DIR}/training-knn.sh" "$@"

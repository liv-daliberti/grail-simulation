#!/usr/bin/env bash
# Wrapper that runs the KNN pipeline for opinion tasks only.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export KNN_PIPELINE_TASKS="opinion"
exec "${SCRIPT_DIR}/training-knn.sh" "$@"

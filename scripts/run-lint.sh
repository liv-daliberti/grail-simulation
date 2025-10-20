#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"

echo "Running pylint on clean_data..."
"${PYTHON_BIN}" -m pylint clean_data "$@"

echo "Running pylint on prompt_builder..."
"${PYTHON_BIN}" -m pylint prompt_builder src/prompt_builder.py "$@"

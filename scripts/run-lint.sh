#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"

echo "Running pylint (errors only) on clean_data and src..."
"${PYTHON_BIN}" -m pylint --rcfile=development/.pylintrc --disable=all --enable=E clean_data src "$@"

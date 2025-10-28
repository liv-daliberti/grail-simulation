#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

# Activate local virtualenv if present
VENV_PATH=${TRAINING_VENV_PATH:-"${ROOT_DIR}/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${ROOT_DIR}/src:${ROOT_DIR}:${PYTHONPATH:-}"

cd "${ROOT_DIR}"

CONFIG_FILE=""
if [[ -f "development/pytest.ini" ]]; then
  CONFIG_FILE="-c development/pytest.ini"
elif [[ -f "pytest.ini" ]]; then
  CONFIG_FILE="-c pytest.ini"
fi

echo "Running pytest..."
"${PYTHON_BIN}" -m pytest ${CONFIG_FILE} "$@"

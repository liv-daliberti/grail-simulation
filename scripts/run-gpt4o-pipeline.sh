#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

VENV_PATH=${TRAINING_VENV_PATH:-"${REPO_ROOT}/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"

OUT_DIR="${GPT4O_OUT_DIR:-${REPO_ROOT}/models/gpt-4o}"
CACHE_DIR="${GPT4O_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/gpt4o}"
REPORTS_DIR="${GPT4O_REPORTS_DIR:-${REPO_ROOT}/reports/gpt4o}"
SWEEP_DIR="${GPT4O_SWEEP_DIR:-${OUT_DIR}/sweeps}"

mkdir -p "${OUT_DIR}" "${CACHE_DIR}" "${REPORTS_DIR}" "${SWEEP_DIR}"

ARGS=(
  --out-dir "${OUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --reports-dir "${REPORTS_DIR}"
  --sweep-dir "${SWEEP_DIR}"
)

if [[ -n "${GPT4O_DATASET:-}" ]]; then
  ARGS+=(--dataset "${GPT4O_DATASET}")
fi
if [[ -n "${GPT4O_ISSUES:-}" ]]; then
  ARGS+=(--issues "${GPT4O_ISSUES}")
fi
if [[ -n "${GPT4O_STUDIES:-}" ]]; then
  ARGS+=(--studies "${GPT4O_STUDIES}")
fi
if [[ -n "${GPT4O_EVAL_MAX:-}" ]]; then
  ARGS+=(--eval-max "${GPT4O_EVAL_MAX}")
fi
if [[ -n "${GPT4O_TEMPERATURE_GRID:-}" ]]; then
  ARGS+=(--temperature-grid "${GPT4O_TEMPERATURE_GRID}")
fi
if [[ -n "${GPT4O_MAX_TOKENS_GRID:-}" ]]; then
  ARGS+=(--max-tokens-grid "${GPT4O_MAX_TOKENS_GRID}")
fi
if [[ -n "${GPT4O_TOP_P_GRID:-}" ]]; then
  ARGS+=(--top-p-grid "${GPT4O_TOP_P_GRID}")
fi
if [[ -n "${GPT4O_LOG_LEVEL:-}" ]]; then
  ARGS+=(--log-level "${GPT4O_LOG_LEVEL}")
fi
if [[ "${GPT4O_OVERWRITE:-0}" != "0" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${GPT4O_DRY_RUN:-0}" != "0" ]]; then
  ARGS+=(--dry-run)
fi

exec "${PYTHON_BIN}" -m gpt4o.pipeline "${ARGS[@]}" "$@"

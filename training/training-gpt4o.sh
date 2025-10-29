#!/usr/bin/env bash
# Run the GPT-4o slate baseline (sweeps + final evaluation + reports).
# Usage: bash training/training-gpt4o.sh [--eval-max N ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/gpt-4o}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/gpt4o}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT_DIR/reports/gpt4o}"
SWEEP_DIR="${GPT4O_SWEEP_DIR:-$OUT_DIR/sweeps}"

VENV_PATH=${TRAINING_VENV_PATH:-"${ROOT_DIR}/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
  else
    echo "[gpt4o] Warning: expected virtualenv at ${VENV_PATH}/bin/activate; continuing without activation." >&2
  fi
fi

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$REPORTS_DIR" "$SWEEP_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"

ARGS=(
  --out-dir "$OUT_DIR"
  --cache-dir "$CACHE_DIR"
  --reports-dir "$REPORTS_DIR"
  --sweep-dir "$SWEEP_DIR"
)

if [[ -n "${DATASET:-}" ]]; then
  ARGS+=(--dataset "$DATASET")
fi
if [[ -n "${ISSUES:-}" ]]; then
  ARGS+=(--issues "$ISSUES")
fi
if [[ -n "${STUDIES:-}" ]]; then
  ARGS+=(--studies "$STUDIES")
fi
if [[ -n "${OPINION_STUDIES:-}" ]]; then
  ARGS+=(--opinion-studies "$OPINION_STUDIES")
fi
if [[ -n "${EVAL_MAX:-}" ]]; then
  ARGS+=(--eval-max "$EVAL_MAX")
fi
if [[ -n "${TEMPERATURE_GRID:-}" ]]; then
  ARGS+=(--temperature-grid "$TEMPERATURE_GRID")
fi
if [[ -n "${MAX_TOKENS_GRID:-}" ]]; then
  ARGS+=(--max-tokens-grid "$MAX_TOKENS_GRID")
fi
if [[ -n "${TOP_P_GRID:-}" ]]; then
  ARGS+=(--top-p-grid "$TOP_P_GRID")
fi
if [[ -n "${OPINION_MAX_PARTICIPANTS:-}" ]]; then
  ARGS+=(--opinion-max-participants "$OPINION_MAX_PARTICIPANTS")
fi
if [[ -n "${OPINION_DIRECTION_TOLERANCE:-}" ]]; then
  ARGS+=(--opinion-direction-tolerance "$OPINION_DIRECTION_TOLERANCE")
fi
if [[ -n "${REQUEST_RETRIES:-}" ]]; then
  ARGS+=(--request-retries "$REQUEST_RETRIES")
fi
if [[ -n "${REQUEST_RETRY_DELAY:-}" ]]; then
  ARGS+=(--request-retry-delay "$REQUEST_RETRY_DELAY")
fi
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  ARGS+=(--overwrite)
fi

python -m gpt4o.pipeline "${ARGS[@]}" "$@"

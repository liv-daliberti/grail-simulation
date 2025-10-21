#!/usr/bin/env bash
# Run the GPT-4o slate baseline (sweeps + final evaluation + reports).
# Usage: bash training/training-gpt4o.sh [--eval-max N ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/gpt4o}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/gpt4o}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT_DIR/reports/gpt4o}"
SWEEP_DIR="${GPT4O_SWEEP_DIR:-$OUT_DIR/sweeps}"

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
if [[ -n "${EVAL_MAX:-}" ]]; then
  ARGS+=(--eval-max "$EVAL_MAX")
fi
if [[ -n "${TEMPERATURE_GRID:-}" ]]; then
  ARGS+=(--temperature-grid "$TEMPERATURE_GRID")
fi
if [[ -n "${MAX_TOKENS_GRID:-}" ]]; then
  ARGS+=(--max-tokens-grid "$MAX_TOKENS_GRID")
fi
if [[ "${OVERWRITE:-0}" != "0" ]]; then
  ARGS+=(--overwrite)
fi

python -m gpt4o.pipeline "${ARGS[@]}" "$@"


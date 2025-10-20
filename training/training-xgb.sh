#!/usr/bin/env bash
# Run the XGBoost slate baseline with optional model export.
# Usage: bash training/training-xgb.sh [--issues minimum_wage ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET="${DATASET:-$ROOT_DIR/data/cleaned_grail}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/xgb}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/xgb}"
MODEL_DIR="${MODEL_DIR:-$OUT_DIR/checkpoints}"

mkdir -p "$OUT_DIR" "$CACHE_DIR"

CLI=(
  "$PYTHON_BIN" "-m" "xgb.cli"
  "--dataset" "$DATASET"
  "--out_dir" "$OUT_DIR"
  "--cache_dir" "$CACHE_DIR"
  "--max_train" "${MAX_TRAIN:-200000}"
  "--max_features" "${MAX_FEATURES:-200000}"
  "--eval_max" "${EVAL_MAX:-0}"
  "--seed" "${SEED:-42}"
  "--xgb_learning_rate" "${XGB_LEARNING_RATE:-0.1}"
  "--xgb_max_depth" "${XGB_MAX_DEPTH:-6}"
  "--xgb_n_estimators" "${XGB_N_ESTIMATORS:-300}"
  "--xgb_subsample" "${XGB_SUBSAMPLE:-0.8}"
  "--xgb_colsample_bytree" "${XGB_COLSAMPLE_BYTREE:-0.8}"
  "--xgb_tree_method" "${XGB_TREE_METHOD:-hist}"
  "--xgb_reg_lambda" "${XGB_REG_LAMBDA:-1.0}"
  "--xgb_reg_alpha" "${XGB_REG_ALPHA:-0.0}"
  "--log_level" "${LOG_LEVEL:-INFO}"
)

if [[ "${FIT_MODEL:-1}" == "1" ]]; then
  CLI+=("--fit_model")
  if [[ "${SAVE_MODEL:-1}" == "1" ]]; then
    mkdir -p "$MODEL_DIR"
    CLI+=("--save_model" "$MODEL_DIR")
  fi
fi

if [[ -n "${LOAD_MODEL:-}" ]]; then
  CLI+=("--load_model" "$LOAD_MODEL")
fi

if [[ -n "${EXTRA_TEXT_FIELDS:-}" ]]; then
  CLI+=("--extra_text_fields" "$EXTRA_TEXT_FIELDS")
fi

if [[ -n "${ISSUES:-}" ]]; then
  CLI+=("--issues" "$ISSUES")
fi

if [[ "${OVERWRITE:-1}" == "1" ]]; then
  CLI+=("--overwrite")
fi

CLI+=("$@")

"${CLI[@]}"

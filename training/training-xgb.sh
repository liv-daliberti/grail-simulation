#!/usr/bin/env bash
# Run the end-to-end XGBoost pipeline (hyperparameter sweeps + reports).
# Usage: bash training/training-xgb.sh [--issues minimum_wage ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
DATASET="${DATASET:-$ROOT_DIR/data/cleaned_grail}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/xgb}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/xgb}"
MODEL_DIR="${MODEL_DIR:-$OUT_DIR/checkpoints}"
REPORTS_DIR="${REPORTS_DIR:-$ROOT_DIR/reports/xgb}"
SWEEP_DIR="${SWEEP_DIR:-$OUT_DIR/sweeps}"

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$REPORTS_DIR" "$MODEL_DIR" "$SWEEP_DIR"

CLI=(
  "$PYTHON_BIN" "-m" "xgb.pipeline"
  "--dataset" "$DATASET"
  "--out-dir" "$OUT_DIR"
  "--cache-dir" "$CACHE_DIR"
  "--reports-dir" "$REPORTS_DIR"
  "--sweep-dir" "$SWEEP_DIR"
  "--save-model-dir" "$MODEL_DIR"
  "--max-train" "${MAX_TRAIN:-200000}"
  "--max-features" "${MAX_FEATURES:-200000}"
  "--eval-max" "${EVAL_MAX:-0}"
  "--seed" "${SEED:-42}"
  "--opinion-max-participants" "${OPINION_MAX_PARTICIPANTS:-0}"
  "--tree-method" "${XGB_TREE_METHOD:-hist}"
  "--learning-rate-grid" "${XGB_LEARNING_RATE_GRID:-0.05,0.1,0.2}"
  "--max-depth-grid" "${XGB_MAX_DEPTH_GRID:-4,6}"
  "--n-estimators-grid" "${XGB_N_ESTIMATORS_GRID:-200,400}"
  "--subsample-grid" "${XGB_SUBSAMPLE_GRID:-0.7,0.9}"
  "--colsample-grid" "${XGB_COLSAMPLE_GRID:-0.7,1.0}"
  "--reg-lambda-grid" "${XGB_REG_LAMBDA_GRID:-1.0}"
  "--reg-alpha-grid" "${XGB_REG_ALPHA_GRID:-0.0,0.5}"
  "--log-level" "${LOG_LEVEL:-INFO}"
)

if [[ -n "${EXTRA_TEXT_FIELDS:-}" ]]; then
  CLI+=("--extra-text-fields" "$EXTRA_TEXT_FIELDS")
fi

if [[ -n "${ISSUES:-}" ]]; then
  CLI+=("--issues" "$ISSUES")
fi

if [[ -n "${STUDIES:-}" ]]; then
  CLI+=("--studies" "$STUDIES")
fi

if [[ "${OVERWRITE:-1}" == "1" ]]; then
  CLI+=("--overwrite")
fi

CLI+=("$@")

"${CLI[@]}"

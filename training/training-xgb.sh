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
TEXT_VECTORIZER_GRID="${TEXT_VECTORIZER_GRID:-tfidf,word2vec,sentence_transformer}"
WORD2VEC_SIZE="${WORD2VEC_SIZE:-256}"
WORD2VEC_WINDOW="${WORD2VEC_WINDOW:-5}"
WORD2VEC_MIN_COUNT="${WORD2VEC_MIN_COUNT:-2}"
WORD2VEC_EPOCHS="${WORD2VEC_EPOCHS:-10}"
WORD2VEC_WORKERS="${WORD2VEC_WORKERS:-1}"
WORD2VEC_MODEL_DIR="${WORD2VEC_MODEL_DIR:-$OUT_DIR/word2vec_models}"
SENTENCE_TRANSFORMER_MODEL="${SENTENCE_TRANSFORMER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
SENTENCE_TRANSFORMER_DEVICE="${SENTENCE_TRANSFORMER_DEVICE:-}"
SENTENCE_TRANSFORMER_BATCH_SIZE="${SENTENCE_TRANSFORMER_BATCH_SIZE:-32}"
SENTENCE_TRANSFORMER_NORMALIZE="${SENTENCE_TRANSFORMER_NORMALIZE:-1}"

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$REPORTS_DIR" "$MODEL_DIR" "$SWEEP_DIR" "$WORD2VEC_MODEL_DIR"

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
  "--text-vectorizer-grid" "$TEXT_VECTORIZER_GRID"
  "--word2vec-size" "$WORD2VEC_SIZE"
  "--word2vec-window" "$WORD2VEC_WINDOW"
  "--word2vec_min_count" "$WORD2VEC_MIN_COUNT"
  "--word2vec-epochs" "$WORD2VEC_EPOCHS"
  "--word2vec-workers" "$WORD2VEC_WORKERS"
  "--word2vec-model-dir" "$WORD2VEC_MODEL_DIR"
  "--sentence-transformer-model" "$SENTENCE_TRANSFORMER_MODEL"
  "--sentence-transformer-batch-size" "$SENTENCE_TRANSFORMER_BATCH_SIZE"
  "--log-level" "${LOG_LEVEL:-INFO}"
)

if [[ -n "$SENTENCE_TRANSFORMER_DEVICE" ]]; then
  CLI+=("--sentence-transformer-device" "$SENTENCE_TRANSFORMER_DEVICE")
fi

if [[ "$SENTENCE_TRANSFORMER_NORMALIZE" == "0" ]]; then
  CLI+=("--sentence-transformer-no-normalize")
else
  CLI+=("--sentence-transformer-normalize")
fi

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

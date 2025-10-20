#!/usr/bin/env bash
# Run the KNN slate baseline for gun rights and minimum wage issues.
# Usage: bash training/training-knn.sh [--eval_max N ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET="${DATASET:-$ROOT_DIR/data/cleaned_grail}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/knn}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/knn}"
WORD2VEC_MODEL_DIR="${WORD2VEC_MODEL_DIR:-$OUT_DIR/word2vec_models}"
KNN_K="${KNN_K:-25}"
KNN_K_SWEEP="${KNN_K_SWEEP:-1,2,3,4,5,10,15,20,25,50}"
KNN_METRIC="${KNN_METRIC:-cosine}"
KNN_MAX_TRAIN="${KNN_MAX_TRAIN:-200000}"
EVAL_MAX="${EVAL_MAX:-0}"
WORD2VEC_SIZE="${WORD2VEC_SIZE:-256}"

ISSUES=("minimum_wage" "gun_control")

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$WORD2VEC_MODEL_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"

run_knn() {
  local feature_space=$1
  local out_subdir=$2
  shift 2

  python "$ROOT_DIR/src/knn/knn-baseline.py" \
    --dataset "$DATASET" \
    --feature-space "$feature_space" \
    --fit-index \
    --knn_k "$KNN_K" \
    --knn_k_sweep "$KNN_K_SWEEP" \
    --knn_metric "$KNN_METRIC" \
    --knn_max_train "$KNN_MAX_TRAIN" \
    --eval_max "$EVAL_MAX" \
    --out_dir "$out_subdir" \
    --cache_dir "$CACHE_DIR" \
    --overwrite \
    "$@"
}

for issue in "${ISSUES[@]}"; do
  run_knn "tfidf" "$OUT_DIR/tfidf" --issues "$issue" "$@"
  run_knn "word2vec" "$OUT_DIR/word2vec" \
    --issues "$issue" \
    --word2vec-size "$WORD2VEC_SIZE" \
    --word2vec-model-dir "$WORD2VEC_MODEL_DIR" \
    "$@"
done

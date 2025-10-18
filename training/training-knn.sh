#!/usr/bin/env bash
# Run the KNN slate baseline for all issues using the cleaned dataset.
# Usage: bash training/training-knn.sh [--eval_max N ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET="${DATASET:-$ROOT_DIR/data/cleaned_grail}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/knn}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/knn}"  # separate cache for TF-IDF runs

mkdir -p "$OUT_DIR" "$CACHE_DIR"

# default args; allow additional CLI flags to be forwarded
python "$ROOT_DIR/src/knn/knn-baseline.py" \
  --dataset "$DATASET" \
  --fit_index \
  --knn_k "${KNN_K:-25}" \
  --knn_metric "${KNN_METRIC:-cosine}" \
  --knn_max_train "${KNN_MAX_TRAIN:-200000}" \
  --eval_max "${EVAL_MAX:-0}" \
  --out_dir "$OUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  --overwrite \
  "$@"

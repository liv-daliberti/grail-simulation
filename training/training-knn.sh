#!/usr/bin/env bash
# Run the KNN slate baseline for gun rights and minimum wage issues.
# Usage: bash training/training-knn.sh [--eval_max N ...]

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
DATASET="${DATASET:-$ROOT_DIR/data/cleaned_grail}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/models/knn}"
CACHE_DIR="${CACHE_DIR:-$ROOT_DIR/.cache/huggingface/knn}"
WORD2VEC_MODEL_DIR="${WORD2VEC_MODEL_DIR:-$OUT_DIR/word2vec_models}"
KNN_K_SWEEP="${KNN_K_SWEEP:-1,2,3,4,5,10,15,20,25,50,100}"
WORD2VEC_EPOCHS="${WORD2VEC_EPOCHS:-10}"

if command -v nproc >/dev/null 2>&1; then
  NUM_CPUS=$(nproc)
elif command -v getconf >/dev/null 2>&1; then
  NUM_CPUS=$(getconf _NPROCESSORS_ONLN)
else
  NUM_CPUS=1
fi
MAX_WORD2VEC_WORKERS="${MAX_WORD2VEC_WORKERS:-40}"
if [ "$NUM_CPUS" -lt 1 ]; then
  NUM_CPUS=1
fi
if [ "$NUM_CPUS" -gt "$MAX_WORD2VEC_WORKERS" ]; then
  WORD2VEC_WORKERS_DEFAULT=$MAX_WORD2VEC_WORKERS
else
  WORD2VEC_WORKERS_DEFAULT=$NUM_CPUS
fi
WORD2VEC_WORKERS="${WORD2VEC_WORKERS:-$WORD2VEC_WORKERS_DEFAULT}"

mkdir -p "$OUT_DIR" "$CACHE_DIR" "$WORD2VEC_MODEL_DIR"

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src"
export WORD2VEC_WORKERS
export WORD2VEC_EPOCHS

SWEEP_DIR="${KNN_SWEEP_DIR:-$OUT_DIR/sweeps}"

python -m knn.pipeline \
  --dataset "$DATASET" \
  --out-dir "$OUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --word2vec-model-dir "$WORD2VEC_MODEL_DIR" \
  --sweep-dir "$SWEEP_DIR" \
  --k-sweep "$KNN_K_SWEEP" \
  "$@"

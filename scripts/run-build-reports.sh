#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"

DATASET="${REPORTS_DATASET:-${REPO_ROOT}/data/cleaned_grail}"
log "Using dataset: ${DATASET}"

if [ ! -d "${DATASET}" ]; then
  case "${DATASET}" in
    */* | *:*)
      log "Dataset path does not exist locally. The pipeline will attempt to load it via datasets.load_dataset."
      ;;
    *)
      log "Dataset directory '${DATASET}' not found. Set REPORTS_DATASET to a valid local path or HF dataset id."
      exit 1
      ;;
  esac
fi

KNN_OUT_DIR="${KNN_REPORTS_OUT_DIR:-${REPO_ROOT}/models/knn}"
KNN_CACHE_DIR="${KNN_REPORTS_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/knn}"
KNN_W2V_DIR="${KNN_REPORTS_WORD2VEC_DIR:-${KNN_OUT_DIR}/word2vec_models}"

mkdir -p "${KNN_OUT_DIR}" "${KNN_CACHE_DIR}" "${KNN_W2V_DIR}"

log "Building KNN reports..."
python -m knn.pipeline \
  --dataset "${DATASET}" \
  --out-dir "${KNN_OUT_DIR}" \
  --cache-dir "${KNN_CACHE_DIR}" \
  --word2vec-model-dir "${KNN_W2V_DIR}"

XGB_OUT_DIR="${XGB_REPORTS_OUT_DIR:-${REPO_ROOT}/models/xgb}"
XGB_CACHE_DIR="${XGB_REPORTS_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/xgb}"
XGB_REPORTS_DIR="${XGB_REPORTS_DIR:-${REPO_ROOT}/reports/xgb}"

mkdir -p "${XGB_OUT_DIR}" "${XGB_CACHE_DIR}" "${XGB_REPORTS_DIR}"

log "Building XGBoost reports..."
python -m xgb.pipeline \
  --dataset "${DATASET}" \
  --out-dir "${XGB_OUT_DIR}" \
  --cache-dir "${XGB_CACHE_DIR}" \
  --reports-dir "${XGB_REPORTS_DIR}"

log "Report builds completed."

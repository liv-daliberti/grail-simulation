#!/usr/bin/env bash
# Submit finalize jobs for the KNN pipeline using cached sweep results.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

ACCOUNT=${ACCOUNT:-mltheory}
PARTITION=${PARTITION:-mltheory}
GRES=${GRES:-gpu:1}
CPUS=${CPUS:-16}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MEMORY=${MEMORY:-128G}
LOG_DIR=${LOG_DIR:-"${ROOT_DIR}/logs/knn"}

mkdir -p "${LOG_DIR}"

COMMON_FLAGS=(
  --account="${ACCOUNT}"
  --partition="${PARTITION}"
  --gres="${GRES}"
  --cpus-per-task="${CPUS}"
  --time="${TIME_LIMIT}"
  --mem="${MEMORY}"
  --export="ALL,TRAINING_REPO_ROOT=${ROOT_DIR}"
)

submit_finalize() {
  local job_name=$1
  shift
  sbatch \
    --job-name="${job_name}" \
    --output="${LOG_DIR}/${job_name}_%j.out" \
    --error="${LOG_DIR}/${job_name}_%j.err" \
    "${COMMON_FLAGS[@]}" \
    "$@"
}

echo "[submit-knn-finalize] Submitting next-video finalize job..."
submit_finalize "knn-next-final" \
  "${SCRIPT_DIR}/training-knn-next.sh" finalize --no-reuse-final "$@"

echo "[submit-knn-finalize] Submitting opinion finalize job..."
submit_finalize "knn-opinion-final" \
  "${SCRIPT_DIR}/training-knn-opinion.sh" finalize --no-reuse-final "$@"

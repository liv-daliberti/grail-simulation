#!/usr/bin/env bash
# Submit finalize jobs for the XGBoost pipeline using cached sweep results.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "${SCRIPT_DIR}/.." && pwd)

ACCOUNT=${ACCOUNT:-mltheory}
PARTITION=${PARTITION:-mltheory}
GRES=${GRES:-gpu:1}
CPUS=${CPUS:-16}
TIME_LIMIT=${TIME_LIMIT:-04:00:00}
MEMORY=${MEMORY:-128G}
LOG_DIR=${LOG_DIR:-"${ROOT_DIR}/logs/xgb"}

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

flag_present() {
  local flag=$1
  shift
  for token in "$@"; do
    if [[ "${token}" == "${flag}" || "${token}" == "${flag}"=* ]]; then
      return 0
    fi
  done
  return 1
}

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

user_args=("$@")
if ! flag_present "--reuse-final" "${user_args[@]}" && ! flag_present "--no-reuse-final" "${user_args[@]}"; then
  user_args+=(--no-reuse-final)
fi
if ! flag_present "--tasks" "${user_args[@]}"; then
  default_tasks=${XGB_FINALIZE_TASKS:-"next_video,opinion"}
  user_args+=(--tasks "${default_tasks}")
fi

echo "[submit-xgb-finalize] Submitting combined finalize job for XGBoost (next-video + opinion)."
submit_finalize "xgb-finalize" \
  "${SCRIPT_DIR}/training-xgb.sh" finalize "${user_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"

DATASET_PATH="${GRAIL_REPORT_DATASET:-${REPO_ROOT}/data/cleaned_grail}"
PROMPT_SAMPLE_TOTAL="${GRAIL_PROMPT_SAMPLE_TOTAL:-10}"
PROMPT_SAMPLE_COUNT="${GRAIL_PROMPT_SAMPLE_COUNT:-1}"
PROMPT_SAMPLE_ISSUES="${GRAIL_PROMPT_SAMPLE_ISSUES:-gun_control,minimum_wage}"
PROMPT_SAMPLE_OUTPUT="${GRAIL_PROMPT_SAMPLE_OUTPUT:-${REPO_ROOT}/reports/prompt_builder/README.md}"

log "Refreshing cleaned dataset, prompt statistics, and replication reports..."
"${REPO_ROOT}/scripts/run-clean-data-suite.sh"

log "Rebuilding recommendation model reports (KNN / XGBoost)..."
"${REPO_ROOT}/scripts/run-build-reports.sh"

log "Regenerating prompt builder samples..."
python -m prompt_builder.samples \
  --dataset "${DATASET_PATH}" \
  --issues "${PROMPT_SAMPLE_ISSUES}" \
  --total "${PROMPT_SAMPLE_TOTAL}" \
  --count "${PROMPT_SAMPLE_COUNT}" \
  --output "${PROMPT_SAMPLE_OUTPUT}"

if [ "${GRAIL_SKIP_GPT4O:-0}" != "1" ]; then
  log "Updating GPT-4o baseline reports..."
  python -m gpt4o.pipeline \
    --dataset "${DATASET_PATH}" \
    --overwrite
else
  log "Skipping GPT-4o report refresh (GRAIL_SKIP_GPT4O=1)."
fi

log "Report refresh completed successfully."

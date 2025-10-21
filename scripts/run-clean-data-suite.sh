#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

SOURCE_DATASET="${GRAIL_SOURCE_DATASET:-}"
if [ -z "${SOURCE_DATASET}" ]; then
  if [ -d "${REPO_ROOT}/capsule-5416997/data" ]; then
    SOURCE_DATASET="capsule-5416997/data"
  else
    log "Set GRAIL_SOURCE_DATASET to a raw dataset (local path or hub id)."
    exit 1
  fi
fi

OUTPUT_DIR="${GRAIL_OUTPUT_DIR:-data/cleaned_grail}"
PROMPT_STATS_DIR="${GRAIL_PROMPT_STATS_DIR:-reports/prompt_stats}"
RESEARCH_REPORT_DIR="${GRAIL_RESEARCH_REPORT_DIR:-reports/research_article_political_sciences}"
HEATMAP_BINS="${GRAIL_RESEARCH_REPORT_BINS:-10}"

PUSH_TO_HUB="${GRAIL_PUSH_TO_HUB:-1}"
HF_TOKEN="${GRAIL_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}}"

DEFAULT_ISSUE_REPOS=("gun_control=od2961/grail-gun" "minimum_wage=od2961/grail-wage")
if [ -n "${GRAIL_ISSUE_REPOS:-}" ]; then
  mapfile -t ISSUE_REPOS < <(printf '%s\n' "${GRAIL_ISSUE_REPOS}")
else
  ISSUE_REPOS=("${DEFAULT_ISSUE_REPOS[@]}")
fi

mkdir -p "${OUTPUT_DIR}" "${PROMPT_STATS_DIR}" "${RESEARCH_REPORT_DIR}"

CLEAN_ARGS=(
  --dataset-name "${SOURCE_DATASET}"
  --output-dir "${OUTPUT_DIR}"
  --prompt-stats-dir "${PROMPT_STATS_DIR}"
)

if [ -n "${GRAIL_TRAIN_SPLIT:-}" ]; then
  CLEAN_ARGS+=(--train-split "${GRAIL_TRAIN_SPLIT}")
fi
if [ -n "${GRAIL_VALIDATION_SPLIT:-}" ]; then
  CLEAN_ARGS+=(--test-split "${GRAIL_VALIDATION_SPLIT}")
fi
if [ -n "${GRAIL_SYSTEM_PROMPT:-}" ]; then
  CLEAN_ARGS+=(--system-prompt "${GRAIL_SYSTEM_PROMPT}")
fi
if [ -n "${GRAIL_SOL_KEY:-}" ]; then
  CLEAN_ARGS+=(--sol-key "${GRAIL_SOL_KEY}")
fi
if [ -n "${GRAIL_MAX_HISTORY:-}" ]; then
  CLEAN_ARGS+=(--max-history "${GRAIL_MAX_HISTORY}")
fi
if [ -n "${GRAIL_VALIDATION_RATIO:-}" ]; then
  CLEAN_ARGS+=(--validation-ratio "${GRAIL_VALIDATION_RATIO}")
fi

if [ "${PUSH_TO_HUB}" = "1" ]; then
  if [ -z "${HF_TOKEN}" ]; then
    log "Set HUGGINGFACE_TOKEN (or GRAIL_HUB_TOKEN) to push cleaned splits."
    exit 1
  fi
  for issue_pair in "${ISSUE_REPOS[@]}"; do
    trimmed="$(echo "${issue_pair}" | xargs)"
    [ -z "${trimmed}" ] && continue
    CLEAN_ARGS+=(--issue-repo "${trimmed}")
  done
  CLEAN_ARGS+=(--push-to-hub --hub-token "${HF_TOKEN}")
fi

log "Running clean_data pipeline..."
python -m clean_data.cli "${CLEAN_ARGS[@]}"

log "Building research article replication report..."
python -m clean_data.research_article_political_sciences.cli \
  --dataset "${OUTPUT_DIR}" \
  --output-dir "${RESEARCH_REPORT_DIR}" \
  --bins "${HEATMAP_BINS}"

log "Clean data suite finished."

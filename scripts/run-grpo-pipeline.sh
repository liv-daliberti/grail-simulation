#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"

VENV_PATH=${TRAINING_VENV_PATH:-"${REPO_ROOT}/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"

DATASET="${GRPO_DATASET:-${REPO_ROOT}/data/cleaned_grail}"
SPLIT="${GRPO_SPLIT:-validation}"
OUT_DIR="${GRPO_OUT_DIR:-${REPO_ROOT}/models/grpo}"
CACHE_DIR="${GRPO_CACHE_DIR:-}"
MODEL_PATH="${GRPO_MODEL:-}"
MODEL_REVISION="${GRPO_MODEL_REVISION:-}"
MODEL_DTYPE="${GRPO_MODEL_DTYPE:-auto}"
LABEL_OVERRIDE="${GRPO_LABEL:-}"
STAGE="${GRPO_STAGE:-full}"

if [[ "${STAGE}" != "reports" && "${STAGE}" != "evaluate" && "${STAGE}" != "full" ]]; then
  echo "Invalid GRPO_STAGE '${STAGE}'. Expected one of: full, evaluate, reports." >&2
  exit 1
fi

if [[ "${STAGE}" != "reports" && -z "${MODEL_PATH}" && "${GRPO_SKIP_EVAL:-0}" == "0" ]]; then
  echo "GRPO_MODEL must be set when running stages that evaluate the model." >&2
  exit 1
fi

LABEL_FLAG=()
if [[ -n "${LABEL_OVERRIDE}" ]]; then
  LABEL_FLAG=(--label "${LABEL_OVERRIDE}")
elif [[ -n "${MODEL_PATH}" ]]; then
  LABEL_FLAG=(--label "$(basename "${MODEL_PATH}" | tr ' /' '__')")
fi

ARGS=(
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --out-dir "${OUT_DIR}"
  --stage "${STAGE}"
)

if [[ -n "${CACHE_DIR}" ]]; then
  ARGS+=(--cache-dir "${CACHE_DIR}")
fi
if [[ -n "${MODEL_PATH}" ]]; then
  ARGS+=(--model "${MODEL_PATH}")
fi
if [[ -n "${MODEL_REVISION}" ]]; then
  ARGS+=(--revision "${MODEL_REVISION}")
fi
if [[ -n "${MODEL_DTYPE}" ]]; then
  ARGS+=(--dtype "${MODEL_DTYPE}")
fi
if [[ -n "${GRPO_SYSTEM_PROMPT_FILE:-}" ]]; then
  ARGS+=(--system-prompt-file "${GRPO_SYSTEM_PROMPT_FILE}")
fi
if [[ -n "${GRPO_OPINION_PROMPT_FILE:-}" ]]; then
  ARGS+=(--opinion-prompt-file "${GRPO_OPINION_PROMPT_FILE}")
fi
if [[ -n "${GRPO_SOLUTION_KEY:-}" ]]; then
  ARGS+=(--solution-key "${GRPO_SOLUTION_KEY}")
fi
if [[ -n "${GRPO_MAX_HISTORY:-}" ]]; then
  ARGS+=(--max-history "${GRPO_MAX_HISTORY}")
fi
if [[ -n "${GRPO_TEMPERATURE:-}" ]]; then
  ARGS+=(--temperature "${GRPO_TEMPERATURE}")
fi
if [[ -n "${GRPO_TOP_P:-}" ]]; then
  ARGS+=(--top-p "${GRPO_TOP_P}")
fi
if [[ -n "${GRPO_MAX_NEW_TOKENS:-}" ]]; then
  ARGS+=(--max-new-tokens "${GRPO_MAX_NEW_TOKENS}")
fi
if [[ -n "${GRPO_EVAL_MAX:-}" ]]; then
  ARGS+=(--eval-max "${GRPO_EVAL_MAX}")
fi
if [[ -n "${GRPO_ISSUES:-}" ]]; then
  ARGS+=(--issues "${GRPO_ISSUES}")
fi
if [[ -n "${GRPO_STUDIES:-}" ]]; then
  ARGS+=(--studies "${GRPO_STUDIES}")
fi
if [[ -n "${GRPO_OPINION_STUDIES:-}" ]]; then
  ARGS+=(--opinion-studies "${GRPO_OPINION_STUDIES}")
fi
if [[ -n "${GRPO_OPINION_MAX_PARTICIPANTS:-}" ]]; then
  ARGS+=(--opinion-max-participants "${GRPO_OPINION_MAX_PARTICIPANTS}")
fi
if [[ -n "${GRPO_DIRECTION_TOLERANCE:-}" ]]; then
  ARGS+=(--direction-tolerance "${GRPO_DIRECTION_TOLERANCE}")
fi
if [[ -n "${GRPO_LOG_LEVEL:-}" ]]; then
  ARGS+=(--log-level "${GRPO_LOG_LEVEL}")
fi
if [[ "${GRPO_OVERWRITE:-0}" != "0" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${GRPO_NO_NEXT_VIDEO:-0}" != "0" ]]; then
  ARGS+=(--no-next-video)
fi
if [[ "${GRPO_NO_OPINION:-0}" != "0" ]]; then
  ARGS+=(--no-opinion)
fi

exec "${PYTHON_BIN}" -m grpo.pipeline "${ARGS[@]}" "${LABEL_FLAG[@]}" "$@"

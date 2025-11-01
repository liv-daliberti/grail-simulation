#!/bin/bash
#SBATCH --job-name=GRPO_EVAL_GUN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=logs/grpo_eval/gun/slurm_%j.out
#SBATCH --export=ALL,LOG_DIR=/n/fs/similarity/grail-simulation/logs/grpo_eval/gun

set -euo pipefail
trap 'rc=$?; echo "[evaluate-grpo-gun] failure (exit ${rc}) at line ${LINENO}" >&2' ERR

# ────────────────────────────────────────────────────────────────
# Resolve repository root
# ────────────────────────────────────────────────────────────────
if [[ -z "${ROOT_DIR:-}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    ROOT_DIR="$SLURM_SUBMIT_DIR"
  else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
  fi
fi
export ROOT_DIR

LOG_DIR=${LOG_DIR_OVERRIDE:-${LOG_DIR:-"$ROOT_DIR/logs/grpo_eval/gun"}}
export LOG_DIR
RUN_LABEL=${RUN_LABEL:-grpo-gun-checkpoint-50}
MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models/grpo/gun/checkpoint-50"}
# Evaluate gun model on the gun validation split only
DATASET=${DATASET:-"$ROOT_DIR/data/cleaned_grail/gun_control"}
SPLIT=${SPLIT:-validation}
OUT_DIR=${OUT_DIR:-"$MODEL_PATH"}
STAGE=${STAGE:-evaluate}
LOG_LEVEL=${LOG_LEVEL:-INFO}
# Restrict to the gun issue by default
ISSUES=${ISSUES:-gun_control}
# Default report location and label per model
REPORTS_SUBDIR=${REPORTS_SUBDIR:-grpo-gun}
BASELINE_LABEL=${BASELINE_LABEL:-"GRPO (Gun)"}

# Do not force stage/label here; override via environment when needed, e.g.:
#   STAGE=full RUN_LABEL=grpo-gun-checkpoint-50 scripts/evaluate-grpo-gun.sh
mkdir -p "$LOG_DIR"

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export PYTHONFAULTHANDLER=${PYTHONFAULTHANDLER:-1}

echo "[evaluate-grpo-gun] $(date --iso-8601=seconds) • starting evaluation; logs -> $LOG_DIR" >&2

if command -v module >/dev/null 2>&1; then
  module load cudatoolkit/12.4 || true
fi

VENV_PATH=${VENV_PATH:-"$ROOT_DIR/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" && -d "$VENV_PATH" ]]; then
  # shellcheck source=/dev/null
  source "$VENV_PATH/bin/activate"
fi

if ! command -v python >/dev/null 2>&1; then
  echo "python binary not found; ensure VENV_PATH is correct or pre-activate an environment." >&2
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:$ROOT_DIR/src:$ROOT_DIR"

# Keep all caches/config under the repo to avoid HOME/TMP quota issues
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$ROOT_DIR/.cache"}
export XDG_CONFIG_HOME=${XDG_CONFIG_HOME:-"$ROOT_DIR/.config"}
export TMPDIR=${TMPDIR:-"$ROOT_DIR/.tmp"}
export HF_HOME=${HF_HOME:-"$ROOT_DIR/.hf_cache"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"$ROOT_DIR/.cache/huggingface/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$ROOT_DIR/.cache/huggingface/datasets"}
export TORCH_HOME=${TORCH_HOME:-"$XDG_CACHE_HOME/torch"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$ROOT_DIR/.triton"}
export VLLM_CONFIG_ROOT=${VLLM_CONFIG_ROOT:-"$XDG_CONFIG_HOME/vllm"}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-"$XDG_CACHE_HOME/vllm"}
export VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH:-"$TMPDIR"}
export VLLM_NO_USAGE_STATS=${VLLM_NO_USAGE_STATS:-1}
export VLLM_DO_NOT_TRACK=${VLLM_DO_NOT_TRACK:-1}
export DO_NOT_TRACK=${DO_NOT_TRACK:-1}
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$TMPDIR" "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$VLLM_CONFIG_ROOT" "$VLLM_CACHE_ROOT"

EXTRA_ARGS=()
[[ -n "${MODEL_DTYPE:-}" ]] && EXTRA_ARGS+=(--dtype "$MODEL_DTYPE")
[[ -n "${MODEL_REVISION:-}" ]] && EXTRA_ARGS+=(--revision "$MODEL_REVISION")
[[ -n "${SYSTEM_PROMPT_FILE:-}" ]] && EXTRA_ARGS+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
[[ -n "${OPINION_PROMPT_FILE:-}" ]] && EXTRA_ARGS+=(--opinion-prompt-file "$OPINION_PROMPT_FILE")
[[ -n "${SOLUTION_KEY:-}" ]] && EXTRA_ARGS+=(--solution-key "$SOLUTION_KEY")
[[ -n "${MAX_HISTORY:-}" ]] && EXTRA_ARGS+=(--max-history "$MAX_HISTORY")
[[ -n "${TEMPERATURE:-}" ]] && EXTRA_ARGS+=(--temperature "$TEMPERATURE")
[[ -n "${TOP_P:-}" ]] && EXTRA_ARGS+=(--top-p "$TOP_P")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && EXTRA_ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "${FLUSH_INTERVAL:-}" ]] && EXTRA_ARGS+=(--flush-interval "$FLUSH_INTERVAL")
[[ -n "${EVAL_MAX:-}" ]] && EXTRA_ARGS+=(--eval-max "$EVAL_MAX")
[[ -n "${STUDIES:-}" ]] && EXTRA_ARGS+=(--studies "$STUDIES")
[[ -n "${OPINION_STUDIES:-}" ]] && EXTRA_ARGS+=(--opinion-studies "$OPINION_STUDIES")
[[ -n "${OPINION_MAX_PARTICIPANTS:-}" ]] && EXTRA_ARGS+=(--opinion-max-participants "$OPINION_MAX_PARTICIPANTS")
[[ -n "${DIRECTION_TOLERANCE:-}" ]] && EXTRA_ARGS+=(--direction-tolerance "$DIRECTION_TOLERANCE")
[[ "${OVERWRITE:-0}" != "0" ]] && EXTRA_ARGS+=(--overwrite)
[[ "${NO_NEXT_VIDEO:-0}" != "0" ]] && EXTRA_ARGS+=(--no-next-video)
[[ "${NO_OPINION:-0}" != "0" ]] && EXTRA_ARGS+=(--no-opinion)

if [[ -n "${ISSUES}" ]]; then
  EXTRA_ARGS+=(--issues "$ISSUES")
fi

echo "[evaluate-grpo-gun] $(date --iso-8601=seconds) • launching python -m grpo.pipeline" >&2
srun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 python -u -m grpo.pipeline \
  --model "$MODEL_PATH" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --label "$RUN_LABEL" \
  --out-dir "$OUT_DIR" \
  --stage "$STAGE" \
  --reports-subdir "$REPORTS_SUBDIR" \
  --baseline-label "$BASELINE_LABEL" \
  --log-level "$LOG_LEVEL" \
  "${EXTRA_ARGS[@]}" \
  "$@"


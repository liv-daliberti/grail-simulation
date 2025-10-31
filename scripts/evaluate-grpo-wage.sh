#!/bin/bash
#SBATCH --job-name=GRPO_EVAL_WAGE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:59:00
#SBATCH --output=logs/grpo_eval/wage/slurm_%j.out
#SBATCH --export=ALL,LOG_DIR=/n/fs/similarity/grail-simulation/logs/grpo_eval/wage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR=${LOG_DIR_OVERRIDE:-"$ROOT_DIR/logs/grpo_eval/wage"}
export LOG_DIR
RUN_LABEL=${RUN_LABEL:-grpo-wage-checkpoint-120}
MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models/grpo/wage/checkpoint-120"}
DATASET=${DATASET:-"$ROOT_DIR/data/cleaned_grail"}
SPLIT=${SPLIT:-validation}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/models/grpo"}
STAGE=${STAGE:-evaluate}
LOG_LEVEL=${LOG_LEVEL:-INFO}
ISSUES=${ISSUES:-minimum_wage}

export STAGE=reports 
export RUN_LABEL=grail-gun-checkpoint-120 
mkdir -p "$LOG_DIR"

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

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 python -m grpo.pipeline \
  --model "$MODEL_PATH" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --label "$RUN_LABEL" \
  --out-dir "$OUT_DIR" \
  --stage "$STAGE" \
  --log-level "$LOG_LEVEL" \
  "${EXTRA_ARGS[@]}" \
  "$@"

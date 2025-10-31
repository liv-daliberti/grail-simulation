#!/bin/bash
#SBATCH --job-name=GRAIL_EVAL_WAGE
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/grail_eval/wage/slurm_%j.out
#SBATCH --export=ALL,LOG_DIR=/n/fs/similarity/grail-simulation/logs/grail_eval/wage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_DIR=${LOG_DIR_OVERRIDE:-"$ROOT_DIR/logs/grail_eval/wage"}
export LOG_DIR
RUN_LABEL=${RUN_LABEL:-grail-wage-checkpoint-120}
MODEL_PATH=${MODEL_PATH:-"$ROOT_DIR/models/grail/wage/checkpoint-120"}
DATASET=${DATASET:-"$ROOT_DIR/data/cleaned_grail/minimum_wage"}
SPLIT=${SPLIT:-validation}
OUT_DIR=${OUT_DIR:-"$ROOT_DIR/models/grail"}
STAGE=${STAGE:-evaluate}
LOG_LEVEL=${LOG_LEVEL:-INFO}

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
[[ -n "${SYSTEM_PROMPT_FILE:-}" ]] && EXTRA_ARGS+=(--system-prompt-file "$SYSTEM_PROMPT_FILE")
[[ -n "${OPINION_PROMPT_FILE:-}" ]] && EXTRA_ARGS+=(--opinion-prompt-file "$OPINION_PROMPT_FILE")
[[ -n "${SOLUTION_KEY:-}" ]] && EXTRA_ARGS+=(--solution-key "$SOLUTION_KEY")
[[ -n "${MAX_HISTORY:-}" ]] && EXTRA_ARGS+=(--max-history "$MAX_HISTORY")
[[ -n "${TEMPERATURE:-}" ]] && EXTRA_ARGS+=(--temperature "$TEMPERATURE")
[[ -n "${TOP_P:-}" ]] && EXTRA_ARGS+=(--top-p "$TOP_P")
[[ -n "${MAX_NEW_TOKENS:-}" ]] && EXTRA_ARGS+=(--max-new-tokens "$MAX_NEW_TOKENS")
[[ -n "${EVAL_MAX:-}" ]] && EXTRA_ARGS+=(--eval-max "$EVAL_MAX")
[[ -n "${ISSUES:-}" ]] && EXTRA_ARGS+=(--issues "$ISSUES")
[[ -n "${STUDIES:-}" ]] && EXTRA_ARGS+=(--studies "$STUDIES")
[[ -n "${OPINION_STUDIES:-}" ]] && EXTRA_ARGS+=(--opinion-studies "$OPINION_STUDIES")
[[ -n "${OPINION_MAX_PARTICIPANTS:-}" ]] && EXTRA_ARGS+=(--opinion-max-participants "$OPINION_MAX_PARTICIPANTS")
[[ -n "${DIRECTION_TOLERANCE:-}" ]] && EXTRA_ARGS+=(--direction-tolerance "$DIRECTION_TOLERANCE")
[[ "${OVERWRITE:-0}" != "0" ]] && EXTRA_ARGS+=(--overwrite)
[[ "${NO_NEXT_VIDEO:-0}" != "0" ]] && EXTRA_ARGS+=(--no-next-video)
[[ "${NO_OPINION:-0}" != "0" ]] && EXTRA_ARGS+=(--no-opinion)

srun --ntasks=1 --gres=gpu:1 --cpus-per-task=8 python -m grail.pipeline \
  --model "$MODEL_PATH" \
  --dataset "$DATASET" \
  --split "$SPLIT" \
  --label "$RUN_LABEL" \
  --out-dir "$OUT_DIR" \
  --stage "$STAGE" \
  --log-level "$LOG_LEVEL" \
  "${EXTRA_ARGS[@]}" \
  "$@"

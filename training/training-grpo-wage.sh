#!/bin/bash
#SBATCH --job-name=GRAIL_GRPO_WAGE
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/grpo_wage/slurm_%j.out
#SBATCH --account=mltheory

set -euo pipefail

# ────────────────────────────────────────────────────────────────
# Core paths (override via environment if needed)
# ────────────────────────────────────────────────────────────────
ROOT_DIR=${ROOT_DIR:-$PWD}
LOG_DIR=${LOG_DIR:-"$ROOT_DIR/logs/grpo_wage"}
RUN_NAME=${RUN_NAME:-Qwen1.5B-GRPO-WAGE}
TIMESTAMP=${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}

CONFIG=${CONFIG:-"$ROOT_DIR/recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_wage.yaml"}
ACCEL_CONFIG=${ACCEL_CONFIG:-"$ROOT_DIR/recipes/accelerate_configs/zero3.yaml"}
MAIN_SCRIPT=${MAIN_SCRIPT:-"$ROOT_DIR/src/grpo/grpo.py"}

mkdir -p "$LOG_DIR"

SERVER_LOG="$LOG_DIR/vllm_${RUN_NAME}_${TIMESTAMP}.log"
TRAINING_LOG="$LOG_DIR/train_${RUN_NAME}_${TIMESTAMP}.log"
ACCEL_CONFIG_TMP=$(mktemp "$LOG_DIR/accelerate_${RUN_NAME}_${TIMESTAMP}_XXXX.yaml")
cleanup_tmp() { rm -f "$ACCEL_CONFIG_TMP"; }
trap cleanup_tmp EXIT
cp "$ACCEL_CONFIG" "$ACCEL_CONFIG_TMP"

GIT_COMMIT=$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_STATUS=$(git -C "$ROOT_DIR" status --short --branch 2>/dev/null || echo "")
export GIT_COMMIT
export GIT_STATUS

# ────────────────────────────────────────────────────────────────
# Environment bootstrap (modules / Conda)
# ────────────────────────────────────────────────────────────────
if command -v module >/dev/null 2>&1; then
  module load cudatoolkit/12.4
fi

CONDA_SH=${CONDA_SH:-/usr/local/anaconda3/2024.02/etc/profile.d/conda.sh}
if [ -f "$CONDA_SH" ]; then
  # shellcheck source=/dev/null
  source "$CONDA_SH"
fi

export CONDA_ENVS_PATH=${CONDA_ENVS_PATH:-"$ROOT_DIR/.conda/envs"}
export CONDA_PKGS_DIRS=${CONDA_PKGS_DIRS:-"$ROOT_DIR/.conda/pkgs"}
export CONDA_CACHE_DIR=${CONDA_CACHE_DIR:-"$ROOT_DIR/.cache/conda"}
export CONDARC=${CONDARC:-"$ROOT_DIR/.condarc"}

mkdir -p "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$CONDA_CACHE_DIR"
cat >"$CONDARC" <<EOF
envs_dirs:
  - $CONDA_ENVS_PATH
pkgs_dirs:
  - $CONDA_PKGS_DIRS
cache_dir: $CONDA_CACHE_DIR
EOF

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_USER=false

# ────────────────────────────────────────────────────────────────
# Cache roots
# ────────────────────────────────────────────────────────────────
export HF_HOME=${HF_HOME:-"$ROOT_DIR/.hf_cache"}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-"$ROOT_DIR/.cache/huggingface/transformers"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$ROOT_DIR/.cache/huggingface/datasets"}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-"$ROOT_DIR/.cache"}
export TMPDIR=${TMPDIR:-"$ROOT_DIR/.tmp"}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-"$ROOT_DIR/.cache/pip"}
export PIP_BUILD_DIR=${PIP_BUILD_DIR:-"$ROOT_DIR/.cache/pip/build"}
export PYTHONPYCACHEPREFIX=${PYTHONPYCACHEPREFIX:-"$ROOT_DIR/.cache/pyc"}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-"$ROOT_DIR/.torchinductor"}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-"$ROOT_DIR/.triton"}

for dir in "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
           "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$LOG_DIR" \
           "$PIP_CACHE_DIR" "$PIP_BUILD_DIR" "$PYTHONPYCACHEPREFIX" "$CONDA_CACHE_DIR"; do
  mkdir -p "$dir"
done

ENV_DIR=${ENV_DIR:-"$CONDA_ENVS_PATH/grail-training"}
PYTHON_VERSION=${PYTHON_VERSION:-3.10}

if [ -d "$ENV_DIR" ]; then
  echo "Removing existing conda env at $ENV_DIR"
  conda env remove -p "$ENV_DIR" -y >/dev/null 2>&1 || rm -rf "$ENV_DIR"
fi

mkdir -p "$(dirname "$ENV_DIR")"
echo "Creating fresh conda env at $ENV_DIR (python $PYTHON_VERSION)"
conda create -y -p "$ENV_DIR" "python=$PYTHON_VERSION"
conda activate "$ENV_DIR"

echo "Python: $(which python)"
python --version

python -m pip install --upgrade pip
python -m pip install -e "$ROOT_DIR/development"
python -m pip install yq

python - <<'PY'
import torch, site, sys
print(f"[torch file] {torch.__file__}")
print(f"[user site] {site.getusersitepackages()}")
print(f"[sys.path head] {sys.path[:5]}")
PY

# ────────────────────────────────────────────────────────────────
# Hugging Face credentials
# ────────────────────────────────────────────────────────────────
if [ -n "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN"
fi
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN="$HF_TOKEN"
fi
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ] && [ -n "${HF_TOKEN:-}" ]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "WARNING: HUGGINGFACE_HUB_TOKEN not set. Push-to-hub will fail." >&2
fi

# ────────────────────────────────────────────────────────────────
# WandB (optional)
# ────────────────────────────────────────────────────────────────
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-"$ROOT_DIR/.wandb"}
export WANDB_ARTIFACT_DIR=${WANDB_ARTIFACT_DIR:-"$WANDB_DIR/artifacts"}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-"$WANDB_DIR/cache"}
export WANDB_CONFIG_DIR=${WANDB_CONFIG_DIR:-"$WANDB_DIR/config"}
export WANDB_PROJECT=${WANDB_PROJECT:-GRAIL}

for dir in "$WANDB_DIR" "$WANDB_ARTIFACT_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"; do
  mkdir -p "$dir"
done

# ────────────────────────────────────────────────────────────────
# Torch / Transformers knobs
# ────────────────────────────────────────────────────────────────
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_LOAD_WEIGHTS_ONLY=${TORCH_LOAD_WEIGHTS_ONLY:-0}
export TORCH_FORCE_FULL_STATE_DICT=${TORCH_FORCE_FULL_STATE_DICT:-1}
export FLASH_ATTENTION_FORCE_DISABLED=${FLASH_ATTENTION_FORCE_DISABLED:-1}
export TRANSFORMERS_NO_FLASH_ATTN=${TRANSFORMERS_NO_FLASH_ATTN:-1}
export VLLM_ATTENTION_BACKEND=${VLLM_ATTENTION_BACKEND:-xformers}
export HF_HUB_REQUEST_TIMEOUT=${HF_HUB_REQUEST_TIMEOUT:-60}
export ACCELERATE_USE_DEEPSPEED_ZERO_INIT=${ACCELERATE_USE_DEEPSPEED_ZERO_INIT:-0}
export DEEPSPEED_ZERO_INIT=${DEEPSPEED_ZERO_INIT:-0}

# ────────────────────────────────────────────────────────────────
# GAIL / GRAIL tuning (override as needed)
# ────────────────────────────────────────────────────────────────
export GAIL_USE=${GAIL_USE:-1}
export GAIL_TRAIN=${GAIL_TRAIN:-1}
export GAIL_WEIGHT=${GAIL_WEIGHT:-0.5}
export GAIL_ALPHA=${GAIL_ALPHA:-1.0}
export GAIL_DISC_MODEL=${GAIL_DISC_MODEL:-distilbert-base-uncased}
export GAIL_LR=${GAIL_LR:-2e-5}
export GRAIL_HISTORY_FULL=${GRAIL_HISTORY_FULL:-0}
export GRAIL_HISTORY_MODE_FULL=${GRAIL_HISTORY_MODE_FULL:-0}
export GRAIL_MAX_HISTORY=${GRAIL_MAX_HISTORY:-12}
export GRAIL_DEBUG=${GRAIL_DEBUG:-0}
export GAIL_EXTRA_NEGS=${GAIL_EXTRA_NEGS:-5}
export GAIL_INSLATE_ONLY=${GAIL_INSLATE_ONLY:-1}
export GAIL_REPLAY=${GAIL_REPLAY:-4096}
export GAIL_MINIBATCH=${GAIL_MINIBATCH:-128}
export GAIL_UPDATES=${GAIL_UPDATES:-2}
export GAIL_DEVICE=${GAIL_DEVICE:-cuda}

# ────────────────────────────────────────────────────────────────
# GPU layout (reserve one GPU for vLLM)
# ────────────────────────────────────────────────────────────────
ALL_GPUS=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
IFS=',' read -r -a GPU_IDS <<< "$ALL_GPUS"
NUM_TOTAL=${#GPU_IDS[@]}
if (( NUM_TOTAL < 2 )); then
  echo "Expected at least two GPUs in CUDA_VISIBLE_DEVICES (got $NUM_TOTAL)" >&2
  exit 1
fi
VLLM_GPU=${GPU_IDS[0]}
TRAINING_GPUS_RAW=("${GPU_IDS[@]:1}")
TRAINING_GPUS=$(printf ",%s" "${TRAINING_GPUS_RAW[@]}")
TRAINING_GPUS=${TRAINING_GPUS#,}
NUM_TRAINING=${#TRAINING_GPUS_RAW[@]}

echo "CUDA_VISIBLE_DEVICES=$ALL_GPUS → reserving GPU $VLLM_GPU for vLLM, $NUM_TRAINING GPUs for training"

python -m yq -y --in-place ".num_processes = $NUM_TRAINING" "$ACCEL_CONFIG_TMP"
echo "Wrote accelerate config override to $ACCEL_CONFIG_TMP"

# ────────────────────────────────────────────────────────────────
# Launch vLLM (1 GPU) + GRPO training (remaining GPUs)
# ────────────────────────────────────────────────────────────────
MAIN_PORT=${MAIN_PORT:-29508}

srun --gres="gpu:${NUM_TOTAL}" --cpus-per-task=64 bash <<BASH
set -euo pipefail

unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_USER=false
export ACCELERATE_LOG_LEVEL=\${ACCELERATE_LOG_LEVEL:-info}
export TORCH_DISTRIBUTED_DEBUG=\${TORCH_DISTRIBUTED_DEBUG:-DETAIL}
export NCCL_DEBUG=\${NCCL_DEBUG:-INFO}
export NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="\$PWD/.triton"
mkdir -p "\$TRITON_CACHE_DIR"

# Spawn vLLM on the reserved GPU
export CUDA_VISIBLE_DEVICES="$VLLM_GPU"
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 2048 --gpu-memory-utilization 0.85 \
  >"$SERVER_LOG" 2>&1 &
VLLM_PID=\$!

for _ in \$(seq 1 90); do
  if curl -sf http://localhost:8000/health >/dev/null; then
    break
  fi
  sleep 2
done

# Launch training on the remaining GPUs
export CUDA_VISIBLE_DEVICES="$TRAINING_GPUS"
TRAIN_EXIT=0
{
  echo "[provenance] git_commit=$GIT_COMMIT"
  if [ -n "$GIT_STATUS" ]; then
    printf '[provenance] git_status\n%s\n' "$GIT_STATUS"
  fi
  python - <<'PY'
import json
import os
from pathlib import Path

info = {
    "config_path": os.environ.get("CONFIG"),
    "dataset_name": None,
    "dataset_revision": None,
    "splits": {},
}

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover - logging only
    info["config_error"] = f"pyyaml_import_failed: {exc}"
    yaml = None

config_path = info["config_path"]
if config_path and yaml is not None and Path(config_path).exists():
    try:
        with open(config_path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle) or {}
        info["dataset_name"] = cfg.get("dataset_name")
    except Exception as exc:  # pragma: no cover - logging only
        info["config_error"] = f"load_failed: {exc}"

dataset_name = info.get("dataset_name")
try:
    from datasets import load_dataset, load_from_disk  # type: ignore
except Exception as exc:  # pragma: no cover - logging only
    info["dataset_error"] = f"datasets_import_failed: {exc}"
else:
    if dataset_name:
        ds_path = Path(dataset_name).expanduser()
        try:
            ds = load_from_disk(str(ds_path)) if ds_path.exists() else load_dataset(dataset_name)
        except Exception as exc:  # pragma: no cover - logging only
            info["dataset_error"] = f"load_failed: {exc}"
        else:
            for split_name, split_ds in ds.items():
                split_info = {
                    "num_rows": len(split_ds),
                    "fingerprint": getattr(split_ds, "_fingerprint", None),
                }
                details = getattr(split_ds, "info", None)
                if details:
                    revision = getattr(details, "dataset_revision", None)
                    if revision:
                        split_info["dataset_revision"] = revision
                        info["dataset_revision"] = info.get("dataset_revision") or revision
                info["splits"][split_name] = split_info

print("[provenance] dataset=" + json.dumps(info, default=str))
PY
  accelerate launch --main_process_port $MAIN_PORT --config_file "$ACCEL_CONFIG_TMP" \
    "$MAIN_SCRIPT" --config "$CONFIG" --use_vllm \
    --run_name "${RUN_NAME}-${TIMESTAMP}" --ignore_data_skip --seed 42
} 2>&1 | tee "$TRAINING_LOG"
TRAIN_EXIT=\${PIPESTATUS[0]}

kill "\$VLLM_PID" >/dev/null 2>&1 || true
wait "\$VLLM_PID" 2>/dev/null || true

exit \$TRAIN_EXIT
BASH

echo "Logs:"
echo "  vLLM:   $SERVER_LOG"
echo "  Train:  $TRAINING_LOG"

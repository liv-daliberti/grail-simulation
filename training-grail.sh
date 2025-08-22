#!/bin/bash
#SBATCH --job-name=GRAIL_GRPO
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm_%j.out

set -euo pipefail

# ──────────────────────────────────────────────────────────────────────
# Paths you may customize
# ──────────────────────────────────────────────────────────────────────
CONFIG="/n/fs/similarity/trees/recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail.yaml"
ACCEL_CONFIG="/n/fs/similarity/trees/recipes/accelerate_configs/zero3.yaml"
MAIN_SCRIPT="/n/fs/similarity/trees/src/open_r1/grpo.py"   # updated entry with GAIL+ReAct

RUN_NAME="Qwen1.5B-GRAIL"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="logs/vllm_${RUN_NAME}_${TIMESTAMP}.log"
TRAINING_LOG="logs/train_${RUN_NAME}_${TIMESTAMP}.log"

# ──────────────────────────────────────────────────────────────────────
# Modules / env
# ──────────────────────────────────────────────────────────────────────
module load cudatoolkit/12.4
source /usr/local/anaconda3/2024.02/etc/profile.d/conda.sh

# Activate your env
ENV_DIR="/n/fs/similarity/open-r1/openr1"
conda activate "$ENV_DIR"
echo "✅ Conda: $(which python)"; python --version

# make absolutely sure user site-packages are ignored
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_USER=false

# ensure pip installs go into the env (not ~/.local)
python -m pip install --no-user -U huggingface_hub yq

# sanity check: torch should come from the conda env
python - <<'PY'
import torch, site, sys
print("[torch file]", torch.__file__)
print("[user site]", site.getusersitepackages())
print("[sys.path head]", sys.path[:5])
PY

# Minimal pip bits
python -m pip install -q --upgrade huggingface_hub yq

# ──────────────────────────────────────────────────────────────────────
# Caches (local workspace)
# ──────────────────────────────────────────────────────────────────────
ROOT_DIR="$PWD"
export HF_HOME="$ROOT_DIR/.hf_cache"
export TRANSFORMERS_CACHE="$ROOT_DIR/.cache/huggingface/transformers"
export HF_DATASETS_CACHE="$ROOT_DIR/.cache/huggingface/datasets"
export XDG_CACHE_HOME="$ROOT_DIR/.cache"
export TMPDIR="$ROOT_DIR/.tmp"
export TORCHINDUCTOR_CACHE_DIR="$ROOT_DIR/.torchinductor"
export TRITON_CACHE_DIR="$ROOT_DIR/.triton"

mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME" \
         "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" logs

# W&B offload dirs (optional)
export WANDB_MODE=online
export WANDB_DIR=/n/fs/similarity/wandb-offload/tmp
export WANDB_ARTIFACT_DIR=/n/fs/similarity/wandb-offload/artifacts
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-offload/cache
export WANDB_CONFIG_DIR=/n/fs/similarity/wandb-offload/config

# Torch settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_LOAD_WEIGHTS_ONLY=0
export TORCH_FORCE_FULL_STATE_DICT=1
export FLASH_ATTENTION_FORCE_DISABLED=1
export TRANSFORMERS_NO_FLASH_ATTN=1

# vLLM settings
export VLLM_ATTENTION_BACKEND=xformers
export VLLM_API_KEY="dummy"
export HF_HUB_REQUEST_TIMEOUT=60
export ACCELERATE_USE_DEEPSPEED_ZERO_INIT=0
export DEEPSPEED_ZERO_INIT=0

# ──────────────────────────────────────────────────────────────────────
# Determine training/vLLM GPU split
# ──────────────────────────────────────────────────────────────────────
ALL_GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_TOTAL=$(echo "$ALL_GPUS" | tr ',' '\n' | wc -l)
NUM_TRAINING=$(( NUM_TOTAL - 1 ))
echo "CUDA_VISIBLE_DEVICES=$ALL_GPUS → training GPUs: $NUM_TRAINING (reserve 1 for vLLM)"

# Update accelerate num_processes to training GPUs
cp -f "$ACCEL_CONFIG" "${ACCEL_CONFIG}.bak"
python -m yq -y --in-place ".num_processes = $NUM_TRAINING" "$ACCEL_CONFIG"
echo "→ accelerate num_processes set to $NUM_TRAINING in $ACCEL_CONFIG"

# provide the new token
export HUGGING_FACE_HUB_TOKEN="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"

# ──────────────────────────────────────────────────────────────────────
# GAIL env toggles (slate-aware discriminator)
# ──────────────────────────────────────────────────────────────────────
export GAIL_USE=1
export GAIL_DISC_MODEL="distilbert-base-uncased"
export GAIL_LR="2e-5"
export GAIL_ALPHA="1.0"
export GRAIL_HISTORY_FULL=0
export GRAIL_HISTORY_MODE_FULL=0
export GRAIL_MAX_HISTORY=12   # any positive number
export GRAIL_DEBUG=1
# makes the disc bite a bit
export GAIL_EXTRA_NEGS=5
export GAIL_INSLATE_ONLY=1

# stabilize the disc
export GAIL_REPLAY=4096
export GAIL_MINIBATCH=128
export GAIL_UPDATES=2

export WANDB_DIR=/n/fs/similarity/wandb
export WANDB_CACHE_DIR=/n/fs/similarity/wandb-cache
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

# keep GAIL off the GPUs to avoid any chance of DS hooks
export GAIL_DEVICE=cuda
export GAIL_OWNER_RANK=$(( NUM_TRAINING - 1 ))   # → 6
export GAIL_DEVICE=cuda:$(( NUM_TRAINING - 1 ))  # → cuda:6 given 7 training GPUs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ──────────────────────────────────────────────────────────────────────
# Launch vLLM (GPU 0) + GRPO training (remaining GPUs)
# ──────────────────────────────────────────────────────────────────────
srun --gres=gpu:$NUM_TOTAL --cpus-per-task=64 bash -c '
set -euo pipefail

# hygiene
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_USER=false
export ACCELERATE_LOG_LEVEL=debug
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR="$PWD/.triton"; mkdir -p "$TRITON_CACHE_DIR"

# vLLM on GPU 0
export CUDA_VISIBLE_DEVICES=$(echo "'"$ALL_GPUS"'" | cut -d"," -f1)
trl vllm-serve --model Qwen/Qwen2.5-1.5B-Instruct --dtype float16 --port 8000 \
  --tensor-parallel-size 1 --max-model-len 2048 --gpu-memory-utilization 0.85 \
  > "'"$SERVER_LOG"'" 2>&1 & VLLM_PID=$!

# wait health
for i in $(seq 1 90); do curl -sf http://localhost:8000/health && break || true; sleep 2; done

# training on remaining GPUs
export CUDA_VISIBLE_DEVICES=$(echo "'"$ALL_GPUS"'" | cut -d"," -f2-)
accelerate launch --main_process_port 29508 --config_file "'"$ACCEL_CONFIG"'" \
  "'"$MAIN_SCRIPT"'" --config "'"$CONFIG"'" --use_vllm \
  --run_name "'"${RUN_NAME}-${TIMESTAMP}"'" --ignore_data_skip --seed 42 \
  > "'"$TRAINING_LOG"'" 2>&1 || true

kill $VLLM_PID || true
wait $VLLM_PID || true
'


echo "Logs:"
echo "  vLLM:   $SERVER_LOG"
echo "  Train:  $TRAINING_LOG"

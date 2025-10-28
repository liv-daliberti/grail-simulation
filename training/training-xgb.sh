#!/usr/bin/env bash
#SBATCH --job-name=xgb-sweeps
#SBATCH --cpus-per-task=1
#SBATCH --time=00:59:00
#SBATCH --output=xgb_%x_%A_%a.out
#SBATCH --error=xgb_%x_%A_%a.err

# Automated SLURM submission wrapper for the XGBoost pipeline.
# Outside SLURM it plans the sweep, submits an array job covering every
# configuration, and optionally schedules a dependent finalize job.

set -euo pipefail

ensure_dual_task_string() {
  local raw="$1"
  local -a tokens=()
  local -a extras=()
  declare -A seen=()
  IFS=',' read -r -a tokens <<<"${raw}"
  for token in "${tokens[@]}"; do
    local trimmed
    trimmed=$(echo "${token}" | tr '[:upper:]' '[:lower:]')
    trimmed=$(echo "${trimmed}" | xargs 2>/dev/null || echo "${trimmed}")
    [[ -z "${trimmed}" ]] && continue
    case "${trimmed}" in
      next|next_video|next-video|nextvideo|slate)
        seen[next_video]=1
        ;;
      opinion|opinion_stage|opinion-stage)
        seen[opinion]=1
        ;;
      *)
        if [[ -z "${seen[${trimmed}]:-}" ]]; then
          extras+=("${trimmed}")
          seen["${trimmed}"]=1
        fi
        ;;
    esac
  done
  local -a ordered=(next_video opinion)
  for extra in "${extras[@]}"; do
    if [[ "${extra}" != "next_video" && "${extra}" != "opinion" ]]; then
      ordered+=("${extra}")
    fi
  done
  printf '%s\n' "$(IFS=','; echo "${ordered[*]}")"
}

SCRIPT_PATH=$(realpath "${BASH_SOURCE[0]}")
if [[ -n "${TRAINING_REPO_ROOT:-}" ]]; then
  ROOT_DIR=$(realpath "${TRAINING_REPO_ROOT}")
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT_DIR=$(realpath "${SLURM_SUBMIT_DIR}")
else
  ROOT_DIR=$(cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)
fi

VENV_PATH=${TRAINING_VENV_PATH:-"${ROOT_DIR}/.venv"}
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -f "${VENV_PATH}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${VENV_PATH}/bin/activate"
  else
    echo "[xgb] Warning: expected virtualenv at ${VENV_PATH}/bin/activate; continuing without activation." >&2
  fi
fi

PYTHON_BIN=${PYTHON_BIN:-python}
DEFAULT_LOG_DIR="${ROOT_DIR}/logs/xgb"
LOG_DIR_CANDIDATE=${LOG_DIR:-${DEFAULT_LOG_DIR}}
if ! mkdir -p "${LOG_DIR_CANDIDATE}" 2>/dev/null; then
  echo "[xgb] Warning: unable to use LOG_DIR='${LOG_DIR_CANDIDATE}'. Falling back to ${DEFAULT_LOG_DIR}." >&2
  LOG_DIR_CANDIDATE="${DEFAULT_LOG_DIR}"
  mkdir -p "${LOG_DIR_CANDIDATE}"
fi
LOG_DIR="${LOG_DIR_CANDIDATE}"

HF_CACHE_DIR_DEFAULT="${ROOT_DIR}/.cache/huggingface/knn"
if [[ -z "${HF_HOME:-}" ]]; then
  export HF_HOME="${HF_CACHE_DIR_DEFAULT}"
fi
if [[ -z "${HF_DATASETS_CACHE:-}" ]]; then
  export HF_DATASETS_CACHE="${HF_CACHE_DIR_DEFAULT}"
fi
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}"


# ---------------------------------------------------------------------------
# Default GPU configuration (overridable via environment variables)
# ---------------------------------------------------------------------------

: "${XGB_USE_GPU:=1}"
: "${XGB_GPU_PARTITION:=gpu}"
: "${XGB_GPU_GRES:=gpu:1}"
: "${XGB_GPU_CPUS:=16}"
: "${XGB_GPU_MEM:=128G}"
: "${XGB_GPU_MAX_ARRAY_SIZE:=1000}"
: "${XGB_SENTENCE_DEVICE:=cuda}"
: "${XGB_REUSE_FINAL:=1}"
: "${XGB_FINAL_USE_GPU:=1}"
: "${XGB_FINAL_PARTITION:=}"
: "${XGB_FINAL_GRES:=}"
: "${XGB_FINAL_CPUS:=}"
: "${XGB_FINAL_MEM:=}"
: "${XGB_FINAL_TIME:=}"
export XGB_REUSE_FINAL
DEFAULT_XGB_PIPELINE_TASKS="next_video,opinion"
: "${XGB_PIPELINE_TASKS:=${DEFAULT_XGB_PIPELINE_TASKS}}"
XGB_PIPELINE_TASKS=$(ensure_dual_task_string "${XGB_PIPELINE_TASKS}")
: "${XGB_LEARNING_RATE_GRID:=0.05}"
: "${XGB_MAX_DEPTH_GRID:=4}"
: "${XGB_N_ESTIMATORS_GRID:=300}"
: "${XGB_SUBSAMPLE_GRID:=0.9}"
: "${XGB_COLSAMPLE_GRID:=0.8}"
: "${XGB_REG_LAMBDA_GRID:=1.0}"
: "${XGB_REG_ALPHA_GRID:=0.0}"
: "${XGB_TEXT_VECTORIZER_GRID:=tfidf,word2vec,sentence_transformer}"

export PYTHONPATH="${ROOT_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
cd "${ROOT_DIR}"

print_usage() {
  cat <<'EOF'
Usage:
  training/training-xgb.sh                    # plan + submit array + finalize
  training/training-xgb.sh plan [...]         # show sweep catalogue only
  training/training-xgb.sh finalize [...]     # run finalize stage locally
  sbatch training/training-xgb.sh sweeps [...]    # internal worker invocation
  sbatch training/training-xgb.sh finalize [...]  # internal dependency job

Environment overrides:
  PYTHON_BIN             Python executable (default: python)
  LOG_DIR                Directory for SLURM outputs (default: logs/xgb)
  XGB_SKIP_FINALIZE      Skip automatic finalize submission when set to 1
  XGB_SWEEP_JOB_NAME     Override sweep job name (default: xgb-sweeps)
  XGB_FINAL_JOB_NAME     Override finalize job name (default: xgb-finalize)
  XGB_SWEEP_TIME         Wallclock time for sweep tasks (default: 00:59:00)
  XGB_FINAL_TIME         Wallclock time for finalize task (default: 06:00:00)
  XGB_FINAL_CPUS         CPU count for finalize task (default: 8)
  XGB_MAX_ARRAY_SIZE     Maximum tasks per SLURM array submission (default: 1000)
  XGB_GPU_FEATURES       Feature spaces that require GPU scheduling (default: * for all)
  XGB_USE_GPU            Enable GPU scheduling when set to 1 (auto-enabled if GPU options are provided)
  XGB_GPU_PARTITION      Partition name for GPU sweep chunks
  XGB_GPU_GRES           GPU resource specification for sweep chunks (e.g. gpu:1)
  XGB_GPU_CPUS           CPU count per GPU sweep task (default: 4)
  XGB_GPU_MEM            Memory request per GPU sweep task
  XGB_GPU_TIME           Wallclock limit for GPU sweep tasks (default: XGB_SWEEP_TIME)
  XGB_GPU_MAX_ARRAY_SIZE Maximum tasks per GPU array submission (default: XGB_MAX_ARRAY_SIZE)
  XGB_GPU_SBATCH_FLAGS   Additional sbatch flags appended to GPU sweep submissions
  XGB_SENTENCE_DEVICE    Overrides SENTENCE_TRANSFORMER_DEVICE when GPU sweeps run (default: cuda)
  XGB_FINAL_USE_GPU      Use GPU resources for the finalize stage when GPU sweeps ran (default: 1)
  XGB_FINAL_PARTITION    Partition name for the finalize stage (defaults to GPU partition when set)
  XGB_FINAL_GRES         GPU resource specification for the finalize stage
  XGB_FINAL_CPUS         CPU count for the finalize stage (defaults to GPU CPU count)
  XGB_FINAL_MEM          Memory request for the finalize stage
  XGB_FINAL_TIME         Wallclock limit for the finalize stage
  XGB_FINAL_SBATCH_FLAGS Additional sbatch flags appended to the finalize submission
EOF
}

append_flag_once() {
  local -n arr_ref=$1
  local flag=$2
  local value=$3
  for ((i = 0; i < ${#arr_ref[@]}; ++i)); do
    if [[ "${arr_ref[i]}" == "${flag}" ]]; then
      return
    fi
  done
  arr_ref+=("${flag}" "${value}")
}

ensure_tasks_flag() {
  local -n target=$1
  local default_value=${2:-"${XGB_PIPELINE_TASKS}"}
  local canonical=""
  for ((i = 0; i < ${#target[@]}; ++i)); do
    if [[ "${target[i]}" == "--tasks" ]]; then
      local existing="${target[i + 1]:-}"
      canonical=$(ensure_dual_task_string "${existing}")
      target[i + 1]="${canonical}"
      break
    fi
  done
  if [[ -z "${canonical}" ]]; then
    canonical=$(ensure_dual_task_string "${default_value}")
    append_flag_once target --tasks "${canonical}"
  fi
  printf '%s\n' "${canonical}"
}

apply_default_sweep_grids() {
  local -n target=$1
  append_flag_once target --learning-rate-grid "${XGB_LEARNING_RATE_GRID}"
  append_flag_once target --max-depth-grid "${XGB_MAX_DEPTH_GRID}"
  append_flag_once target --n-estimators-grid "${XGB_N_ESTIMATORS_GRID}"
  append_flag_once target --subsample-grid "${XGB_SUBSAMPLE_GRID}"
  append_flag_once target --colsample-grid "${XGB_COLSAMPLE_GRID}"
  append_flag_once target --reg-lambda-grid "${XGB_REG_LAMBDA_GRID}"
  append_flag_once target --reg-alpha-grid "${XGB_REG_ALPHA_GRID}"
  append_flag_once target --text-vectorizer-grid "${XGB_TEXT_VECTORIZER_GRID}"
}

# Return 0 if GPU boosters are supported by the currently installed xgboost.
_xgb_gpu_supported() {
  "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1 || return 1
try:
    import xgboost as _xgb  # type: ignore
    core = _xgb.core  # type: ignore[attr-defined]
    maybe = getattr(core, "_has_cuda_support", None)
    if callable(maybe):
        supported = bool(maybe())
    else:
        lib = getattr(core, "_LIB", None)
        supported = hasattr(lib, "XGBoosterPredictFromDeviceDMatrix")
    raise SystemExit(0 if supported else 1)
except Exception:
    raise SystemExit(1)
PY
}

# Append --tree-method hist when GPU is disabled or unsupported and the flag
# is not already present in the CLI args array passed by name.
ensure_tree_method_default() {
  local -n arr_ref=$1
  # If user already provided an explicit tree method, respect it.
  for ((i = 0; i < ${#arr_ref[@]}; ++i)); do
    if [[ "${arr_ref[i]}" == "--tree-method" ]]; then
      return
    fi
  done
  # Determine if GPU is desired by env (default on) and supported by xgboost.
  local want_gpu=1
  case "${XGB_USE_GPU:-1}" in
    0|false|no|off|disabled) want_gpu=0 ;;
  esac
  if (( want_gpu )); then
    if ! _xgb_gpu_supported; then
      echo "[xgb] XGBoost without GPU support detected; forcing --tree-method hist." >&2
      arr_ref+=("--tree-method" "hist")
    fi
  else
    arr_ref+=("--tree-method" "hist")
  fi
}

check_python_env() {
  # Build required module list based on configured feature spaces.
  local -a required=(datasets xgboost sklearn)
  local text_grid="${XGB_TEXT_VECTORIZER_GRID:-tfidf,word2vec,sentence_transformer}"
  if [[ ",${text_grid}," == *",sentence_transformer,"* ]]; then
    required+=(sentence_transformers)
  fi
  local -a missing=()
  for module in "${required[@]}"; do
    if ! "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
import importlib
import sys
try:
    importlib.import_module("${module}")
except ModuleNotFoundError:
    sys.exit(1)
PY
    then
      missing+=("${module}")
    fi
  done
  # Best-effort auto-install for xgboost only (others still reported).
  local -a remaining=()
  for mod in "${missing[@]}"; do
    if [[ "${mod}" == "xgboost" ]]; then
      echo "[xgb] xgboost missing; attempting pip install ..." >&2
      if [[ -n "${VIRTUAL_ENV:-}" ]]; then
        if ! "${PYTHON_BIN}" -m pip install -q --upgrade xgboost >/dev/null 2>&1; then
          echo "[xgb] Warning: failed to install xgboost into active venv." >&2
        fi
      else
        if ! "${PYTHON_BIN}" -m pip install -q --user --upgrade xgboost >/dev/null 2>&1; then
          echo "[xgb] Warning: failed to install xgboost with --user." >&2
        fi
      fi
      # Re-check import after attempted install.
      if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import sys
import importlib
sys.exit(0 if importlib.util.find_spec('xgboost') is not None else 1)
PY
      then
        remaining+=("xgboost")
      fi
    else
      remaining+=("${mod}")
    fi
  done
  if (( ${#remaining[@]} > 0 )); then
    echo "[xgb] Missing Python modules: ${remaining[*]}" >&2
    echo "[xgb] Install them in your environment, e.g.:" >&2
    echo "       ${PYTHON_BIN} -m pip install ${remaining[*]}" >&2
    exit 1
  fi
}

warm_sentence_transformer() {
  if [[ "${XGB_WARM_SENTENCE_MODEL:-1}" != "1" ]]; then
    return
  fi
  local model_name="${SENTENCE_TRANSFORMER_MODEL:-sentence-transformers/all-mpnet-base-v2}"
  if [[ -z "${model_name}" ]]; then
    return
  fi
  if [[ -n "${XGB_WARM_SENTENCE_MODEL_ONCE:-}" ]]; then
    return
  fi
  echo "[xgb] Warming SentenceTransformer cache for ${model_name} ..."
  if "${PYTHON_BIN}" - <<PY >/dev/null 2>&1
from sentence_transformers import SentenceTransformer
SentenceTransformer("${model_name}")
PY
  then
    export XGB_WARM_SENTENCE_MODEL_ONCE=1
    echo "[xgb] SentenceTransformer cache ready (${model_name})."
  else
    echo "[xgb] Warning: unable to warm SentenceTransformer cache for ${model_name}. Continuing anyway." >&2
  fi
}

append_range_chunks() {
  local start=$1
  local end=$2
  local chunk_size=$3
  local -n target=$4
  if [[ -z "${start}" || -z "${end}" ]]; then
    return
  fi
  if (( chunk_size <= 0 )); then
    chunk_size=1
  fi
  local chunk_start=$start
  while (( chunk_start <= end )); do
    local chunk_end=$(( chunk_start + chunk_size - 1 ))
    if (( chunk_end > end )); then
      chunk_end=$end
    fi
    if (( chunk_start == chunk_end )); then
      target+=("${chunk_start}")
    else
      target+=("${chunk_start}-${chunk_end}")
    fi
    chunk_start=$((chunk_end + 1))
  done
}

parse_range_bounds() {
  local range=$1
  if [[ "${range}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
    RANGE_START=$((10#${BASH_REMATCH[1]}))
    RANGE_END=$((10#${BASH_REMATCH[2]}))
  else
    RANGE_START=$((10#${range}))
    RANGE_END=$RANGE_START
  fi
}

format_range_bounds() {
  local start=$1
  local end=$2
  if (( start > end )); then
    echo ""
  elif (( start == end )); then
    echo "${start}"
  else
    echo "${start}-${end}"
  fi
}

ensure_reuse_final_flag() {
  local -n target=$1
  for flag in "${target[@]}"; do
    if [[ "${flag}" == "--reuse-final" || "${flag}" == "--no-reuse-final" ]]; then
      return
    fi
  done
  target+=("--reuse-final")
}

run_plan() {
  check_python_env
  local -a args=("$@")
  ensure_tree_method_default args
  ensure_reuse_final_flag args
  apply_default_sweep_grids args
  ensure_tasks_flag args "${XGB_PIPELINE_TASKS}"
  "${PYTHON_BIN}" -m xgb.pipeline --stage plan "${args[@]}"
}

submit_jobs() {
  local -a pipeline_args=("$@")
  ensure_reuse_final_flag pipeline_args
  apply_default_sweep_grids pipeline_args
  local pipeline_tasks_raw
  pipeline_tasks_raw=$(ensure_tasks_flag pipeline_args "${XGB_PIPELINE_TASKS}")
  IFS=',' read -r -a pipeline_task_tokens <<<"${pipeline_tasks_raw}"
  local want_next_video=0
  local want_opinion=0
  for token in "${pipeline_task_tokens[@]}"; do
    local trimmed
    trimmed=$(echo "${token}" | tr '[:upper:]' '[:lower:]' | xargs 2>/dev/null)
    [[ -z "${trimmed}" ]] && continue
    case "${trimmed}" in
      next_video|next-video|nextvideo|slate)
        want_next_video=1
        ;;
      opinion|opinion_stage)
        want_opinion=1
        ;;
    esac
  done
  if (( want_next_video == 0 && want_opinion == 0 )); then
    want_next_video=1
    want_opinion=1
  fi

  local plan_output
  plan_output=$(run_plan "${pipeline_args[@]}")
  local total_tasks
  local reuse_only=0
  total_tasks=$(awk -F= '/^TOTAL_TASKS=/{print $2}' <<<"${plan_output}")
  if [[ -z "${total_tasks}" ]]; then
    echo "[xgb] Failed to parse TOTAL_TASKS from plan output." >&2
    exit 1
  fi
  total_tasks=$((10#${total_tasks}))
  if (( total_tasks <= 0 )); then
    echo "[xgb] Plan reported ${total_tasks} tasks; reusing cached sweep metrics."
    reuse_only=1
  fi

  local sweep_job_name="${XGB_SWEEP_JOB_NAME:-xgb-sweeps}"
  local final_job_name="${XGB_FINAL_JOB_NAME:-xgb-finalize}"
  local sweep_time="${XGB_SWEEP_TIME:-00:59:00}"
  local finalize_time="${XGB_FINAL_TIME:-00:59:00}"
  local finalize_cpus="${XGB_FINAL_CPUS:-16}"
  local max_array_size_raw="${XGB_MAX_ARRAY_SIZE:-1000}"
  local slurm_partition=""
  local plan_log="${LOG_DIR}/${sweep_job_name}_plan.txt"

  printf '%s\n' "${plan_output}" > "${plan_log}"
  echo "[xgb] Sweep plan saved to ${plan_log}."
  if ! command -v sbatch >/dev/null 2>&1; then
    echo "[xgb] sbatch not found in PATH. Submit the recorded plan manually or rerun inside a SLURM environment." >&2
    exit 1
  fi
  if (( total_tasks > 1024 )); then
    echo "[xgb] Full sweep plan written to ${plan_log} (showing first 20 rows):"
    head -n 20 "${plan_log}"
  else
    cat "${plan_log}"
  fi

  local max_array_size
  if [[ "${max_array_size_raw}" =~ ^[0-9]+$ ]] && (( max_array_size_raw > 0 )); then
    max_array_size=${max_array_size_raw}
  else
    echo "[xgb] Warning: invalid XGB_MAX_ARRAY_SIZE='${max_array_size_raw}'. Using 1000." >&2
    max_array_size=1000
  fi
  local gpu_max_array_size_raw="${XGB_GPU_MAX_ARRAY_SIZE:-${max_array_size}}"
  local gpu_max_array_size
  if [[ "${gpu_max_array_size_raw}" =~ ^[0-9]+$ ]] && (( gpu_max_array_size_raw > 0 )); then
    gpu_max_array_size=${gpu_max_array_size_raw}
  else
    echo "[xgb] Warning: invalid XGB_GPU_MAX_ARRAY_SIZE='${gpu_max_array_size_raw}'. Using ${max_array_size}." >&2
    gpu_max_array_size=${max_array_size}
  fi

  local gpu_feature_list=${XGB_GPU_FEATURES:-*}
  local gpu_match_all=0
  declare -A GPU_FEATURE_MAP=()
  if [[ "${gpu_feature_list}" == "*" || "${gpu_feature_list,,}" == "all" ]]; then
    gpu_match_all=1
  else
    IFS=',' read -r -a gpu_feature_tokens <<<"${gpu_feature_list}"
    for token in "${gpu_feature_tokens[@]}"; do
      token=$(echo "${token}" | tr '[:upper:]' '[:lower:]' | xargs)
      if [[ -n "${token}" ]]; then
        GPU_FEATURE_MAP["${token}"]=1
      fi
    done
  fi

  local gpu_enabled=${XGB_USE_GPU:-}
  if [[ -z "${gpu_enabled}" ]]; then
    gpu_enabled=1
  fi

  local plan_lines
  plan_lines=$(printf '%s\n' "${plan_output}" | awk -F'	' '$1 ~ /^[0-9]+$/ {print}')
  local current_type=""
  local range_start=""
  local prev_idx=""
  local global_idx=0
  local -a cpu_ranges=()
  local -a gpu_ranges=()
  while IFS=$'	' read -r idx study issue feature label; do
    [[ -z "${idx}" ]] && continue
    local idx_num=${global_idx}
    global_idx=$((global_idx + 1))
    local feature_lower
    feature_lower=$(echo "${feature}" | tr '[:upper:]' '[:lower:]')
    local type="cpu"
    if (( gpu_enabled )); then
      local is_gpu_task=0
      if (( gpu_match_all )); then
        is_gpu_task=1
      elif [[ "${feature_lower}" == "opinion" ]]; then
        if [[ -n "${GPU_FEATURE_MAP[opinion]:-}" ]]; then
          is_gpu_task=1
        fi
      elif [[ -n "${GPU_FEATURE_MAP[${feature_lower}]:-}" ]]; then
        is_gpu_task=1
      fi
      if (( is_gpu_task )); then
        type="gpu"
      fi
    fi
    if [[ "${current_type}" == "${type}" && -n "${range_start}" && idx_num -eq $((prev_idx + 1)) ]]; then
      prev_idx=${idx_num}
    else
      if [[ -n "${range_start}" ]]; then
        if [[ "${current_type}" == "gpu" ]]; then
          append_range_chunks "${range_start}" "${prev_idx}" "${gpu_max_array_size}" gpu_ranges
        else
          append_range_chunks "${range_start}" "${prev_idx}" "${max_array_size}" cpu_ranges
        fi
      fi
      current_type="${type}"
      range_start=${idx_num}
      prev_idx=${idx_num}
    fi
  done <<<"${plan_lines}"
  if [[ -n "${range_start}" ]]; then
    if [[ "${current_type}" == "gpu" ]]; then
      append_range_chunks "${range_start}" "${prev_idx}" "${gpu_max_array_size}" gpu_ranges
    else
      append_range_chunks "${range_start}" "${prev_idx}" "${max_array_size}" cpu_ranges
    fi
  fi

  local -a sweep_job_ids=()
  local cpu_chunk_index=0
  local -a cpu_pending=()
  if (( ! reuse_only )); then
    cpu_pending=("${cpu_ranges[@]}")
  fi
  while (( ${#cpu_pending[@]} > 0 )); do
    local range="${cpu_pending[0]}"
    cpu_pending=("${cpu_pending[@]:1}")
    parse_range_bounds "${range}"
    local chunk_start=${RANGE_START}
    local chunk_end=${RANGE_END}
    if (( chunk_start > chunk_end )); then
      continue
    fi
    local chunk_len=$((chunk_end - chunk_start + 1))
    local chunk_offset=${chunk_start}
    local array_spec
    if (( chunk_len <= 1 )); then
      array_spec="0"
    else
      array_spec="0-$((chunk_len - 1))"
    fi
    local export_vars="ALL,TRAINING_REPO_ROOT=${ROOT_DIR},TRAINING_SWEEP_TOTAL=${total_tasks},TRAINING_SWEEP_OFFSET=${chunk_offset}"
    local sbatch_cmd=(
      sbatch
      --parsable
      --export="${export_vars}"
      --job-name="${sweep_job_name}"
      --array="${array_spec}"
      --time="${sweep_time}"
      --cpus-per-task=1
      --output="${LOG_DIR}/${sweep_job_name}_%A_%a.out"
      --error="${LOG_DIR}/${sweep_job_name}_%A_%a.err"
    )
    sbatch_cmd+=("${SCRIPT_PATH}" sweeps "${pipeline_args[@]}")
    local sbatch_output
    if sbatch_output=$("${sbatch_cmd[@]}" 2>&1); then
      local chunk_job_id="${sbatch_output//$'\n'/}"
      sweep_job_ids+=("${chunk_job_id}")
      echo "[xgb] Submitted sweeps chunk ${cpu_chunk_index} (range ${range}, offset ${chunk_offset}) as job ${chunk_job_id} (array ${array_spec})."
      ((cpu_chunk_index++))
    else
      local trimmed="${sbatch_output//$'\n'/ }"
      trimmed=$(echo "${trimmed}" | xargs 2>/dev/null || echo "${trimmed}")
      if (( chunk_len <= 1 )); then
        echo "[xgb] ERROR: CPU submission failed for index ${chunk_start}: ${trimmed}" >&2
        exit 1
      fi
      echo "[xgb] Warning: CPU chunk ${chunk_start}-${chunk_end} failed (${trimmed}). Bisecting." >&2
      local mid=$(((chunk_start + chunk_end) / 2))
      local first_range
      first_range=$(format_range_bounds "${chunk_start}" "${mid}")
      local second_range
      second_range=$(format_range_bounds $((mid + 1)) "${chunk_end}")
      if [[ -n "${second_range}" ]]; then
        cpu_pending=("${second_range}" "${cpu_pending[@]}")
      fi
      if [[ -n "${first_range}" ]]; then
        cpu_pending=("${first_range}" "${cpu_pending[@]}")
      fi
    fi
  done

  local gpu_chunk_index=0
  if (( gpu_enabled )) && (( ${#gpu_ranges[@]} > 0 )) && (( ! reuse_only )); then
    local gpu_job_name="${sweep_job_name}-gpu"
    local gpu_partition="${XGB_GPU_PARTITION:-gpu}"
    local gpu_gres="${XGB_GPU_GRES:-gpu:1}"
    local gpu_nodes="${XGB_GPU_NODES:-1}"
    local gpu_cpus="${XGB_GPU_CPUS:-16}"
    local gpu_time="${XGB_GPU_TIME:-${sweep_time}}"
    local gpu_mem="${XGB_GPU_MEM:-128G}"
    local gpu_device="${XGB_SENTENCE_DEVICE:-${SENTENCE_TRANSFORMER_DEVICE:-cuda}}"
    local -a extra_gpu_flags=()
    if [[ -n "${XGB_GPU_SBATCH_FLAGS:-}" ]]; then
      read -r -a extra_gpu_flags <<<"${XGB_GPU_SBATCH_FLAGS}"
    fi
    local -a gpu_pending=("${gpu_ranges[@]}")
    while (( ${#gpu_pending[@]} > 0 )); do
      local range="${gpu_pending[0]}"
      gpu_pending=("${gpu_pending[@]:1}")
      if (( ! gpu_enabled )); then
        echo "[xgb] ERROR: GPU execution disabled unexpectedly; aborting." >&2
        exit 1
      fi
      parse_range_bounds "${range}"
      local chunk_start=${RANGE_START}
      local chunk_end=${RANGE_END}
      if (( chunk_start > chunk_end )); then
        continue
      fi
      local chunk_len=$((chunk_end - chunk_start + 1))
      local chunk_offset=${chunk_start}
      local array_spec
      if (( chunk_len <= 1 )); then
        array_spec="0"
      else
        array_spec="0-$((chunk_len - 1))"
      fi
      local export_vars="ALL,TRAINING_REPO_ROOT=${ROOT_DIR},TRAINING_SWEEP_TOTAL=${total_tasks},TRAINING_SWEEP_OFFSET=${chunk_offset},SENTENCE_TRANSFORMER_DEVICE=${gpu_device}"
      local sbatch_cmd=(
        sbatch
        --parsable
        --export="${export_vars}"
        --job-name="${gpu_job_name}"
        --array="${array_spec}"
        --time="${gpu_time}"
        --nodes="${gpu_nodes}"
        --cpus-per-task="${gpu_cpus}"
        --output="${LOG_DIR}/${gpu_job_name}_%A_%a.out"
        --error="${LOG_DIR}/${gpu_job_name}_%A_%a.err"
      )
      if [[ -n "${gpu_partition}" ]]; then
        sbatch_cmd+=(--partition "${gpu_partition}")
      fi
      if [[ -n "${gpu_gres}" ]]; then
        sbatch_cmd+=(--gres "${gpu_gres}")
      fi
      if [[ -n "${gpu_mem}" ]]; then
        sbatch_cmd+=(--mem "${gpu_mem}")
      fi
      if (( ${#extra_gpu_flags[@]} )); then
        sbatch_cmd+=("${extra_gpu_flags[@]}")
      fi
      sbatch_cmd+=("${SCRIPT_PATH}" sweeps "${pipeline_args[@]}")
      local sbatch_output
      if sbatch_output=$("${sbatch_cmd[@]}" 2>&1); then
        local chunk_job_id="${sbatch_output//$'\n'/}"
        sweep_job_ids+=("${chunk_job_id}")
        echo "[xgb] Submitted GPU sweeps chunk ${gpu_chunk_index} (range ${range}, offset ${chunk_offset}) as job ${chunk_job_id} (array ${array_spec})."
        ((gpu_chunk_index++))
      else
        local trimmed="${sbatch_output//$'\n'/ }"
        trimmed=$(echo "${trimmed}" | xargs 2>/dev/null || echo "${trimmed}")
        if (( chunk_len <= 1 )); then
          echo "[xgb] ERROR: GPU submission failed for index ${chunk_start}: ${trimmed}" >&2
          exit 1
        fi
        echo "[xgb] Warning: GPU chunk ${chunk_start}-${chunk_end} failed (${trimmed}). Bisecting." >&2
        local mid=$(((chunk_start + chunk_end) / 2))
        local first_range
        first_range=$(format_range_bounds "${chunk_start}" "${mid}")
        local second_range
        second_range=$(format_range_bounds $((mid + 1)) "${chunk_end}")
        if [[ -n "${second_range}" ]]; then
          gpu_pending=("${second_range}" "${gpu_pending[@]}")
        fi
        if [[ -n "${first_range}" ]]; then
          gpu_pending=("${first_range}" "${gpu_pending[@]}")
        fi
      fi
    done
  fi

  if [[ "${XGB_SKIP_FINALIZE:-0}" == "1" ]]; then
    echo "[xgb] XGB_SKIP_FINALIZE=1; finalize stage not submitted automatically."
    return
  fi

  local dependency_spec=""
  if (( ${#sweep_job_ids[@]} > 0 )); then
    dependency_spec="afterok:${sweep_job_ids[*]}"
    dependency_spec=${dependency_spec// /:}
  elif (( reuse_only )); then
    echo "[xgb] No new sweeps required; scheduling finalize stage to reuse cached metrics."
  else
    echo "[xgb] No sweep jobs were submitted; finalize stage skipped." >&2
    return
  fi

  local finalize_export="ALL,TRAINING_REPO_ROOT=${ROOT_DIR},TRAINING_SWEEP_TOTAL=${total_tasks}"
  local finalize_partition=""
  local finalize_gres=""
  local finalize_nodes="${XGB_FINAL_NODES:-}"
  local finalize_mem="${XGB_FINAL_MEM:-}"
  local default_gpu_partition="${XGB_GPU_PARTITION:-${slurm_partition}}"
  local default_gpu_gres="${XGB_GPU_GRES:-gpu:1}"
  local default_gpu_nodes="${XGB_GPU_NODES:-1}"
  local default_gpu_cpus="${XGB_GPU_CPUS:-16}"
  local default_gpu_time="${XGB_GPU_TIME:-${finalize_time}}"
  local should_use_gpu_finalize=0
  if (( gpu_enabled )) && [[ "${XGB_FINAL_USE_GPU:-1}" == "1" ]] && (( want_next_video || want_opinion )); then
    should_use_gpu_finalize=1
  fi
  if (( should_use_gpu_finalize )); then
    finalize_export+=",SENTENCE_TRANSFORMER_DEVICE=${XGB_SENTENCE_DEVICE:-${SENTENCE_TRANSFORMER_DEVICE:-cuda}}"
    finalize_partition="${XGB_FINAL_PARTITION:-${default_gpu_partition}}"
    finalize_gres="${XGB_FINAL_GRES:-${default_gpu_gres}}"
    finalize_cpus="${XGB_FINAL_CPUS:-${default_gpu_cpus}}"
    finalize_time="${XGB_FINAL_TIME:-${default_gpu_time}}"
    finalize_nodes="${XGB_FINAL_NODES:-${default_gpu_nodes}}"
    if [[ -z "${finalize_mem}" && -n "${XGB_GPU_MEM:-}" ]]; then
      finalize_mem="${XGB_GPU_MEM}"
    fi
  else
    finalize_time="${XGB_FINAL_TIME:-${finalize_time}}"
    finalize_cpus="${XGB_FINAL_CPUS:-${finalize_cpus}}"
    if [[ -z "${finalize_partition}" && -n "${slurm_partition}" ]]; then
      finalize_partition="${slurm_partition}"
    fi
  fi
  if [[ -z "${finalize_partition}" && -n "${slurm_partition}" ]]; then
    finalize_partition="${slurm_partition}"
  fi
  local -a extra_finalize_flags=()
  if [[ -n "${XGB_FINAL_SBATCH_FLAGS:-}" ]]; then
    read -r -a extra_finalize_flags <<<"${XGB_FINAL_SBATCH_FLAGS}"
  fi
  local finalize_cmd=(
    sbatch
    --parsable
    --job-name="${final_job_name}"
    --time="${finalize_time}"
    --cpus-per-task="${finalize_cpus}"
    --output="${LOG_DIR}/${final_job_name}_%A.out"
    --error="${LOG_DIR}/${final_job_name}_%A.err"
    --export="${finalize_export}"
  )
  if [[ -n "${dependency_spec}" ]]; then
    finalize_cmd+=(--dependency "${dependency_spec}")
  fi
  if [[ -n "${finalize_partition}" ]]; then
    finalize_cmd+=(--partition "${finalize_partition}")
  fi
  if [[ -n "${finalize_gres}" ]]; then
    finalize_cmd+=(--gres "${finalize_gres}")
  fi
  if [[ -n "${finalize_nodes}" ]]; then
    finalize_cmd+=(--nodes "${finalize_nodes}")
  fi
  if [[ -n "${finalize_mem}" ]]; then
    finalize_cmd+=(--mem "${finalize_mem}")
  fi
  if (( ${#extra_finalize_flags[@]} )); then
    finalize_cmd+=("${extra_finalize_flags[@]}")
  fi
  finalize_cmd+=("${SCRIPT_PATH}" finalize "${pipeline_args[@]}")
  local finalize_job_id
  finalize_job_id=$("${finalize_cmd[@]}")
  if [[ -n "${dependency_spec}" ]]; then
    echo "[xgb] Finalize job ${finalize_job_id} scheduled with dependency after jobs: ${sweep_job_ids[*]}."
  else
    echo "[xgb] Finalize job ${finalize_job_id} scheduled (no dependencies; reusing cached sweeps)."
  fi
}

run_finalize_local() {
  local -a args=("$@")
  ensure_reuse_final_flag args
  apply_default_sweep_grids args
  echo "[xgb] Running finalize stage locally."
  check_python_env
  ensure_tree_method_default args
  ensure_tasks_flag args "${XGB_PIPELINE_TASKS}"
  "${PYTHON_BIN}" -m xgb.pipeline --stage finalize "${args[@]}"
}

run_sweeps_worker() {
  local -a args=("$@")
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" || "${SLURM_ARRAY_TASK_ID}" == "4294967294" ]]; then
    echo "[xgb] SLURM_ARRAY_TASK_ID missing or control task detected; skipping sweeps helper invocation." >&2
    exit 0
  fi
  local offset_value=0
  if [[ "${TRAINING_SWEEP_OFFSET:-}" =~ ^[0-9]+$ ]]; then
    offset_value=$((10#${TRAINING_SWEEP_OFFSET}))
  fi
  local task_local=$((10#${SLURM_ARRAY_TASK_ID}))
  local task_global=$((offset_value + task_local))
  local sweep_args=(--sweep-task-id "${task_global}")
  if [[ -n "${TRAINING_SWEEP_TOTAL:-}" ]]; then
    sweep_args+=(--sweep-task-count "${TRAINING_SWEEP_TOTAL}")
  elif [[ -n "${SLURM_ARRAY_TASK_COUNT:-}" ]]; then
    sweep_args+=(--sweep-task-count "${SLURM_ARRAY_TASK_COUNT}")
  elif [[ -n "${SLURM_ARRAY_TASK_MAX:-}" && -n "${SLURM_ARRAY_TASK_MIN:-}" ]]; then
    local count=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
    sweep_args+=(--sweep-task-count "${count}")
  fi
  check_python_env
  apply_default_sweep_grids args
  ensure_tree_method_default args
  ensure_tasks_flag args "${XGB_PIPELINE_TASKS}"
  "${PYTHON_BIN}" -m xgb.pipeline --stage sweeps "${sweep_args[@]}" "${args[@]}"
}

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  warm_sentence_transformer
  case "${1:-submit}" in
    submit)
      shift || true
      submit_jobs "$@"
      ;;
    plan)
      shift || true
      run_plan "$@"
      ;;
    finalize)
      shift || true
      run_finalize_local "$@"
      ;;
    sweeps)
      echo "[xgb] Use 'sbatch ${SCRIPT_PATH} sweeps ...' to run worker tasks." >&2
      exit 1
      ;;
    --help|-h)
      print_usage
      ;;
    *)
      submit_jobs "$@"
      ;;
  esac
  exit 0
fi

stage="${1:-sweeps}"
shift || true
case "${stage}" in
  sweeps)
    run_sweeps_worker "$@"
    ;;
  finalize)
    run_finalize_local "$@"
    ;;
  plan)
    run_plan "$@"
    ;;
  *)
    echo "[xgb] Unknown stage '${stage}'." >&2
    print_usage >&2
    exit 1
    ;;
esac

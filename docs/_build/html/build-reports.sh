#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN=${PYTHON_BIN:-python}

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"

DATASET="${REPORTS_DATASET:-${REPO_ROOT}/data/cleaned_grail}"
log "Using dataset target: ${DATASET}"
export REPORTS_DATASET="${DATASET}"
ALLOW_INCOMPLETE_RAW="${REPORTS_ALLOW_INCOMPLETE:-1}"
ALLOW_INCOMPLETE_NORMALIZED="$(printf '%s' "${ALLOW_INCOMPLETE_RAW}" | tr '[:upper:]' '[:lower:]')"
case "${ALLOW_INCOMPLETE_NORMALIZED}" in
  1|true|yes)
    ALLOW_INCOMPLETE="1"
    ;;
  0|false|no)
    ALLOW_INCOMPLETE="0"
    ;;
  *)
    ALLOW_INCOMPLETE="1"
    log "REPORTS_ALLOW_INCOMPLETE='${ALLOW_INCOMPLETE_RAW}' not recognised; defaulting to allow-incomplete mode."
    ;;
esac
KNN_ALLOW_FLAG="--allow-incomplete"
XGB_ALLOW_FLAG="--allow-incomplete"
if [ "${ALLOW_INCOMPLETE}" != "1" ]; then
  KNN_ALLOW_FLAG="--no-allow-incomplete"
  XGB_ALLOW_FLAG="--no-allow-incomplete"
fi
if [ -z "${KNN_ALLOW_INCOMPLETE-}" ]; then
  export KNN_ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE}"
fi
if [ -z "${XGB_ALLOW_INCOMPLETE-}" ]; then
  export XGB_ALLOW_INCOMPLETE="${ALLOW_INCOMPLETE}"
fi

if [ -d "${DATASET}" ]; then
  log "Found local dataset directory."
elif [[ "${DATASET}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]+)?$ ]]; then
  log "Dataset will be loaded from the Hugging Face Hub id '${DATASET}'."
else
  if [ -n "${REPORTS_ISSUE_DATASETS:-}" ]; then
    log "Local dataset missing. Will attempt to assemble it from issue datasets: ${REPORTS_ISSUE_DATASETS}"
    "${PYTHON_BIN}" - <<'PY'
import json
import os
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "pandas is required to assemble REPORTS_ISSUE_DATASETS; install it via pip."
    ) from exc

from datasets import DatasetDict, Dataset, load_dataset

issue_env = os.environ.get("REPORTS_ISSUE_DATASETS", "")
dataset_path = Path(os.environ["REPORTS_DATASET"])
dataset_path.parent.mkdir(parents=True, exist_ok=True)

pairs = []
for raw in issue_env.split(","):
    raw = raw.strip()
    if not raw:
        continue
    if "=" not in raw:
        raise SystemExit(f"Invalid REPORTS_ISSUE_DATASETS entry '{raw}'. Expected issue=dataset_id format.")
    issue, dataset_spec = [token.strip() for token in raw.split("=", 1)]
    if not issue or not dataset_spec:
        raise SystemExit(f"Invalid REPORTS_ISSUE_DATASETS entry '{raw}'.")
    if "#" in dataset_spec:
        dataset_id, config_name = dataset_spec.split("#", 1)
        dataset_id = dataset_id.strip()
        config_name = config_name.strip()
    else:
        dataset_id, config_name = dataset_spec, None
    if not dataset_id:
        raise SystemExit(f"Invalid dataset id in '{raw}'.")
    pairs.append((issue, dataset_id, config_name))

if not pairs:
    raise SystemExit("REPORTS_ISSUE_DATASETS provided but no valid entries were found.")

frames_by_split: dict[str, list[pd.DataFrame]] = {}
for issue, dataset_id, config_name in pairs:
    try:
        load_kwargs = {}
        if config_name:
            load_kwargs["name"] = config_name
        ds = load_dataset(dataset_id, **load_kwargs)
    except Exception as exc:  # pragma: no cover - user feedback
        raise SystemExit(
            f"Failed to load dataset '{dataset_id}'"
            f"{' (config='+config_name+')' if config_name else ''}: {exc}"
        ) from exc
    for split_name, split_ds in ds.items():
        df = split_ds.to_pandas()
        if "issue" not in df.columns:
            df["issue"] = issue
        else:
            df["issue"] = df["issue"].fillna(issue)
        frames_by_split.setdefault(split_name, []).append(df)

if not frames_by_split:
    raise SystemExit("No splits were loaded from the provided datasets.")

merged_splits: dict[str, Dataset] = {}
for split_name, frames in frames_by_split.items():
    combined_df = pd.concat(frames, ignore_index=True, sort=False)
    merged_splits[split_name] = Dataset.from_pandas(combined_df, preserve_index=False)

merged = DatasetDict(merged_splits)
merged.save_to_disk(str(dataset_path))
metadata = {
    "sources": [
        {"issue": issue, "dataset": dataset_id, "config": config_name}
        for issue, dataset_id, config_name in pairs
    ],
    "note": "Assembled by reports/build-reports.sh from issue-level datasets."
}
with open(dataset_path / "_assembly_metadata.json", "w", encoding="utf-8") as handle:
    json.dump(metadata, handle, indent=2)
PY
    log "Dataset assembled at ${DATASET}"
  else
    log "Dataset directory '${DATASET}' not found."
    log "Set REPORTS_DATASET to an existing local path or a Hugging Face dataset id (e.g. user/repo),"
    log "or provide REPORTS_ISSUE_DATASETS (e.g. 'gun_control=user/gun,minimum_wage=user/wage') so it can be assembled automatically."
    exit 1
  fi
fi

KNN_OUT_DIR="${KNN_REPORTS_OUT_DIR:-${REPO_ROOT}/models/knn}"
KNN_CACHE_DIR="${KNN_REPORTS_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/knn}"
KNN_W2V_DIR="${KNN_REPORTS_WORD2VEC_DIR:-${KNN_OUT_DIR}/next_video/word2vec_models}"

mkdir -p "${KNN_OUT_DIR}" "${KNN_CACHE_DIR}" "${KNN_W2V_DIR}"

# Drop any previously generated KNN reports so the regeneration always starts from a clean slate.
KNN_REPORTS_BASE="$("${PYTHON_BIN}" - <<PY
from pathlib import Path
from knn.evaluate import resolve_reports_dir

out_dir = Path(r"${KNN_OUT_DIR}")
print(resolve_reports_dir(out_dir))
PY
)"
KNN_REPORTS_DIR="${KNN_REPORTS_BASE}/knn"
if [ -d "${KNN_REPORTS_DIR}" ]; then
  log "Clearing existing KNN reports at ${KNN_REPORTS_DIR}"
  rm -rf "${KNN_REPORTS_DIR}"
fi
mkdir -p "${KNN_REPORTS_BASE}"

: "${KNN_FEATURE_SPACES:=tfidf,word2vec,sentence_transformer}"
: "${KNN_K_SWEEP:=1,2,3,4,5,10,25,50}"
: "${KNN_PIPELINE_TASKS:=next_video,opinion}"
declare -a KNN_SWEEP_FLAGS=(
  "--feature-spaces" "${KNN_FEATURE_SPACES}"
  "--k-sweep" "${KNN_K_SWEEP}"
  "--tasks" "${KNN_PIPELINE_TASKS}"
)

log "Regenerating KNN reports from existing artefacts..."
if ! find "${KNN_OUT_DIR}" -name "knn_eval_*_validation_metrics.json" -print -quit | grep -q .; then
  if [ "${ALLOW_INCOMPLETE}" = "1" ]; then
    log "No KNN evaluation metrics found under ${KNN_OUT_DIR}. Generating placeholder reports because REPORTS_ALLOW_INCOMPLETE=1."
  else
    log "No KNN evaluation metrics found under ${KNN_OUT_DIR}."
    log "Run 'training/training-knn.sh [pipeline args]' to regenerate sweeps/finalize outputs before rebuilding reports."
    exit 1
  fi
fi
"${PYTHON_BIN}" -m knn.pipeline \
  --dataset "${DATASET}" \
  --out-dir "${KNN_OUT_DIR}" \
  --cache-dir "${KNN_CACHE_DIR}" \
  --word2vec-model-dir "${KNN_W2V_DIR}" \
  "${KNN_SWEEP_FLAGS[@]}" \
  --stage reports \
  "${KNN_ALLOW_FLAG}"

# Drop any previously generated XGBoost reports so the regeneration always starts from a clean slate.
XGB_OUT_DIR="${XGB_REPORTS_OUT_DIR:-${REPO_ROOT}/models/xgb}"
XGB_CACHE_DIR="${XGB_REPORTS_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/xgb}"
XGB_REPORTS_DIR="${XGB_REPORTS_DIR:-${REPO_ROOT}/reports/xgb}"
if [ -d "${XGB_REPORTS_DIR}" ]; then
  log "Clearing existing XGBoost reports at ${XGB_REPORTS_DIR}"
  rm -rf "${XGB_REPORTS_DIR}"
fi

mkdir -p "${XGB_OUT_DIR}" "${XGB_CACHE_DIR}" "${XGB_REPORTS_DIR}"

log "Regenerating XGBoost reports from existing artefacts..."
if ! find "${XGB_OUT_DIR}" -name "metrics.json" -path "*/next_video/*" -print -quit | grep -q .; then
  if [ "${ALLOW_INCOMPLETE}" = "1" ]; then
    log "No XGBoost evaluation metrics found under ${XGB_OUT_DIR}. Generating placeholder reports because REPORTS_ALLOW_INCOMPLETE=1."
  else
    log "No XGBoost evaluation metrics found under ${XGB_OUT_DIR}."
    log "Run 'training/training-xgb.sh [pipeline args]' to regenerate sweeps/finalize outputs before rebuilding reports."
    exit 1
  fi
fi

# Align the report stage CLI with the sweep configuration so cached artefacts are reused.
: "${XGB_LEARNING_RATE_GRID:=0.03,0.06}"
: "${XGB_MAX_DEPTH_GRID:=3,4}"
: "${XGB_N_ESTIMATORS_GRID:=250,350}"
: "${XGB_SUBSAMPLE_GRID:=0.8,0.9}"
: "${XGB_COLSAMPLE_GRID:=0.7}"
: "${XGB_REG_LAMBDA_GRID:=0.7}"
: "${XGB_REG_ALPHA_GRID:=0.1}"
: "${XGB_TEXT_VECTORIZER_GRID:=tfidf,word2vec,sentence_transformer}"
declare -a XGB_SWEEP_FLAGS=(
  "--learning-rate-grid" "${XGB_LEARNING_RATE_GRID}"
  "--max-depth-grid" "${XGB_MAX_DEPTH_GRID}"
  "--n-estimators-grid" "${XGB_N_ESTIMATORS_GRID}"
  "--subsample-grid" "${XGB_SUBSAMPLE_GRID}"
  "--colsample-grid" "${XGB_COLSAMPLE_GRID}"
  "--reg-lambda-grid" "${XGB_REG_LAMBDA_GRID}"
  "--reg-alpha-grid" "${XGB_REG_ALPHA_GRID}"
  "--text-vectorizer-grid" "${XGB_TEXT_VECTORIZER_GRID}"
)
"${PYTHON_BIN}" -m xgb.pipeline \
  --dataset "${DATASET}" \
  --out-dir "${XGB_OUT_DIR}" \
  --cache-dir "${XGB_CACHE_DIR}" \
  --reports-dir "${XGB_REPORTS_DIR}" \
  --stage reports \
  "${XGB_SWEEP_FLAGS[@]}" \
  "${XGB_ALLOW_FLAG}"

GPT4O_OUT_DIR="${GPT4O_REPORTS_OUT_DIR:-${REPO_ROOT}/models/gpt-4o}"
GPT4O_CACHE_DIR="${GPT4O_REPORTS_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/gpt4o}"
GPT4O_REPORTS_DIR="${GPT4O_REPORTS_DIR:-${REPO_ROOT}/reports/gpt4o}"
GPT4O_SWEEP_ROOT="${GPT4O_REPORTS_SWEEP_DIR:-${GPT4O_OUT_DIR}/sweeps}"

log "Regenerating GPT-4o reports from existing artefacts..."
SKIP_GPT4O="0"
if [ ! -d "${GPT4O_SWEEP_ROOT}" ] || ! find "${GPT4O_SWEEP_ROOT}" -name "metrics.json" -print -quit | grep -q .; then
  if [ "${ALLOW_INCOMPLETE}" = "1" ]; then
    log "No GPT-4o sweep metrics found under ${GPT4O_SWEEP_ROOT}. Skipping GPT-4o report refresh because REPORTS_ALLOW_INCOMPLETE=1."
    SKIP_GPT4O="1"
  else
    log "No GPT-4o sweep metrics found under ${GPT4O_SWEEP_ROOT}."
    log "Run 'training/training-gpt4o.sh [pipeline args]' to populate sweeps before rebuilding reports."
    exit 1
  fi
fi

if [ "${SKIP_GPT4O}" != "1" ]; then
  if [ -d "${GPT4O_REPORTS_DIR}" ]; then
    log "Clearing existing GPT-4o reports at ${GPT4O_REPORTS_DIR}"
    rm -rf "${GPT4O_REPORTS_DIR}"
  fi

  declare -a GPT4O_ARGS=(
    "--out-dir" "${GPT4O_OUT_DIR}"
    "--cache-dir" "${GPT4O_CACHE_DIR}"
    "--reports-dir" "${GPT4O_REPORTS_DIR}"
    "--stage" "reports"
  )
  if [ -n "${GPT4O_REPORTS_SWEEP_DIR:-}" ]; then
    GPT4O_ARGS+=("--sweep-dir" "${GPT4O_REPORTS_SWEEP_DIR}")
  fi

  "${PYTHON_BIN}" -m gpt4o.pipeline "${GPT4O_ARGS[@]}"
fi

run_rlhf_report_pipeline() {
  local flavor="$1"
  local module="$2"
  local models_dir="$3"
  local reports_dir="$4"
  local label_override="$5"
  local scenario_name="$6"

  # Prefer checkpoint-50 tree when present (user wants reporting from checkpoint-50)
  local effective_models_dir="${models_dir}"
  if [ -d "${models_dir}/checkpoint-50" ]; then
    effective_models_dir="${models_dir}/checkpoint-50"
  fi

  local next_root="${effective_models_dir}/next_video"
  local opinion_root="${effective_models_dir}/opinion"
  local label="${label_override}"

  if [ -z "${label}" ]; then
    # Try to auto-detect from next_video label directories first
    if [ -d "${next_root}" ]; then
      mapfile -t _labels < <(find "${next_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
    else
      _labels=()
    fi
    if [ -z "${label}" ] && [ ${#_labels[@]} -eq 1 ]; then
      label="${_labels[0]}"
    fi
    # Fallback: detect from opinion if next_video not present or ambiguous
    if [ -z "${label}" ] && [ -d "${opinion_root}" ]; then
      mapfile -t _op_labels < <(find "${opinion_root}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
      if [ ${#_op_labels[@]} -eq 1 ]; then
        label="${_op_labels[0]}"
      fi
    fi
  fi

  local scenario_desc="${scenario_name:-$(basename "${models_dir}")}"
  if [ -z "${scenario_desc}" ] || [ "${scenario_desc}" = "." ] || [ "${scenario_desc}" = "default" ]; then
    scenario_desc="${flavor,,}-default"
  fi

  if [ -z "${label}" ]; then
    log "Skipping ${flavor} report regeneration for scenario ${scenario_desc} (set ${flavor}_REPORT_LABEL or populate ${effective_models_dir})."
    return 1
  fi

  if [ -d "${reports_dir}" ]; then
    log "Clearing existing ${flavor} reports at ${reports_dir}"
    rm -rf "${reports_dir}"
  fi
  mkdir -p "${reports_dir}"

  log "Regenerating ${flavor} reports for label ${label} (scenario ${scenario_desc})"
  declare -a args=(
    "--dataset" "${DATASET}"
    "--out-dir" "${effective_models_dir}"
    "--label" "${label}"
    "--stage" "reports"
  )
  if [ ! -d "${next_root}/${label}" ]; then
    args+=("--no-next-video")
  fi
  if [ ! -d "${opinion_root}/${label}" ]; then
    args+=("--no-opinion")
  fi
  "${PYTHON_BIN}" -m "${module}" "${args[@]}"
}

process_rlhf_family() {
  local flavor="$1"
  local module="$2"
  local base_models_dir="$3"
  local base_reports_dir="$4"
  local label_override="$5"

  local candidates=("${base_models_dir}")
  if [ -d "${base_models_dir}" ]; then
    while IFS= read -r subdir; do
      candidates+=("${base_models_dir}/${subdir}")
    done < <(find "${base_models_dir}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  fi

  local attempted=0
  local succeeded=0
  for candidate in "${candidates[@]}"; do
    if [ ! -d "${candidate}" ]; then
      continue
    fi
    attempted=1
    local scenario_name
    local target_reports_dir="${base_reports_dir}"
    if [ "${candidate}" = "${base_models_dir}" ]; then
      scenario_name="default"
    else
      scenario_name="${candidate##${base_models_dir}/}"
      target_reports_dir="${base_reports_dir}/${scenario_name}"
    fi
    if run_rlhf_report_pipeline "${flavor}" "${module}" "${candidate}" "${target_reports_dir}" "${label_override}" "${scenario_name}"; then
      succeeded=1
    fi
  done

  if [ "${attempted}" -eq 0 ]; then
    log "No ${flavor} artefacts found under ${base_models_dir}; skipping."
  elif [ "${succeeded}" -eq 0 ]; then
    log "Unable to regenerate ${flavor} reports; see messages above for details."
  fi
}

GRPO_MODELS_DIR="${GRPO_MODELS_DIR:-${REPO_ROOT}/models/grpo}"
GRPO_REPORTS_DIR="${GRPO_REPORTS_DIR:-${REPO_ROOT}/reports/grpo}"
GRPO_LABEL="${GRPO_REPORT_LABEL:-}"
process_rlhf_family "GRPO" "grpo.pipeline" "${GRPO_MODELS_DIR}" "${GRPO_REPORTS_DIR}" "${GRPO_LABEL}"

GRAIL_MODELS_DIR="${GRAIL_MODELS_DIR:-${REPO_ROOT}/models/grail}"
GRAIL_REPORTS_DIR="${GRAIL_REPORTS_DIR:-${REPO_ROOT}/reports/grail}"
GRAIL_LABEL="${GRAIL_REPORT_LABEL:-}"
process_rlhf_family "GRAIL" "grail.pipeline" "${GRAIL_MODELS_DIR}" "${GRAIL_REPORTS_DIR}" "${GRAIL_LABEL}"

log "Generating main portfolio comparison report..."
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
from common.reports.portfolio import generate_portfolio_report

repo_root = Path(__file__).resolve().parents[1]
generate_portfolio_report(repo_root)
print(f"[build-reports] Main report written to {repo_root / 'reports' / 'main' / 'README.md'}")
PY

log "Report refresh completed."

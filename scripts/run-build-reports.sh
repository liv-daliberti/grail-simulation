#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

log() {
  printf '[%(%Y-%m-%dT%H:%M:%S%z)T] %s\n' -1 "$*" >&2
}

export PYTHONPATH="${PYTHONPATH:-}:${REPO_ROOT}/src:${REPO_ROOT}"

DATASET="${REPORTS_DATASET:-${REPO_ROOT}/data/cleaned_grail}"
log "Using dataset target: ${DATASET}"
export REPORTS_DATASET="${DATASET}"

if [ -d "${DATASET}" ]; then
  log "Found local dataset directory."
elif [[ "${DATASET}" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(:[A-Za-z0-9_.-]+)?$ ]]; then
  log "Dataset will be loaded from the Hugging Face Hub id '${DATASET}'."
else
  if [ -n "${REPORTS_ISSUE_DATASETS:-}" ]; then
    log "Local dataset missing. Will attempt to assemble it from issue datasets: ${REPORTS_ISSUE_DATASETS}"
    python - <<'PY'
import os
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset

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
    issue, dataset_id = [token.strip() for token in raw.split("=", 1)]
    if not issue or not dataset_id:
        raise SystemExit(f"Invalid REPORTS_ISSUE_DATASETS entry '{raw}'.")
    pairs.append((issue, dataset_id))

if not pairs:
    raise SystemExit("REPORTS_ISSUE_DATASETS provided but no valid entries were found.")

combined: dict[str, list] = {}
for issue, dataset_id in pairs:
    ds = load_dataset(dataset_id)
    for split_name, split_ds in ds.items():
        if "issue" not in split_ds.column_names:
            split_ds = split_ds.add_column("issue", [issue] * len(split_ds))
        combined.setdefault(split_name, []).append(split_ds)

merged = DatasetDict(
    {
        split_name: concatenate_datasets(splits)
        for split_name, splits in combined.items()
    }
)
merged.save_to_disk(str(dataset_path))
metadata = {
    "sources": pairs,
    "note": "Assembled by scripts/run-build-reports.sh from issue-level datasets."
}
with open(dataset_path / "_assembly_metadata.json", "w", encoding="utf-8") as handle:
    import json
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
KNN_W2V_DIR="${KNN_REPORTS_WORD2VEC_DIR:-${KNN_OUT_DIR}/word2vec_models}"

mkdir -p "${KNN_OUT_DIR}" "${KNN_CACHE_DIR}" "${KNN_W2V_DIR}"

log "Building KNN reports..."
python -m knn.pipeline \
  --dataset "${DATASET}" \
  --out-dir "${KNN_OUT_DIR}" \
  --cache-dir "${KNN_CACHE_DIR}" \
  --word2vec-model-dir "${KNN_W2V_DIR}"

XGB_OUT_DIR="${XGB_REPORTS_OUT_DIR:-${REPO_ROOT}/models/xgb}"
XGB_CACHE_DIR="${XGB_REPORTS_CACHE_DIR:-${REPO_ROOT}/.cache/huggingface/xgb}"
XGB_REPORTS_DIR="${XGB_REPORTS_DIR:-${REPO_ROOT}/reports/xgb}"

mkdir -p "${XGB_OUT_DIR}" "${XGB_CACHE_DIR}" "${XGB_REPORTS_DIR}"

log "Building XGBoost reports..."
python -m xgb.pipeline \
  --dataset "${DATASET}" \
  --out-dir "${XGB_OUT_DIR}" \
  --cache-dir "${XGB_CACHE_DIR}" \
  --reports-dir "${XGB_REPORTS_DIR}"

log "Report builds completed."

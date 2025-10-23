# Training Launchers

Shell helpers for running the baseline models and reinforcement-learning jobs
described in the project README.

## Scripts

- `training-grpo.sh` – launches the GRPO baseline with the ZeRO-3 accelerate
  profile and optional discriminator disabled.
- `training-grail.sh` – extends `training-grpo.sh` with the discriminator reward
  path used in the full GRAIL experiments.
- `training-knn.sh` – SLURM array wrapper for the kNN slate baseline
  (exposes `plan`, `sweeps`, and `finalize` helpers).
- `training-xgb.sh` – SLURM array wrapper for the XGBoost slate + opinion
  baseline (same helper workflow as above).

## Usage

Both launchers orchestrate the full SLURM workflow when invoked outside of an
active SLURM job:

```bash
# Plan, submit the sweep array, and queue a finalize job automatically.
training/training-knn.sh --issues minimum_wage

# Same pattern for the XGBoost baseline.
training/training-xgb.sh --issues minimum_wage --text-vectorizer-grid tfidf
```

Under the hood the scripts:

1. Run `--stage plan` to enumerate sweep combinations (printed to stdout).
2. Submit an array job covering every combination.
3. Schedule a dependent job that executes `--stage finalize` (skip with
   `KNN_SKIP_FINALIZE=1` / `XGB_SKIP_FINALIZE=1`).

Logs land under `${LOG_DIR:-logs/knn}` for the kNN launcher and
`${LOG_DIR:-logs/xgb}` for the XGBoost launcher. You can override these paths
via the `LOG_DIR` environment variable (see `--help` output). Each submission
also writes the full sweep catalog to `${LOG_DIR}/<job-name>_plan.txt` for
later inspection.

Manual entry points remain available:

- `training/training-knn.sh plan [...]` – print the sweep catalog without
  submitting jobs.
- `training/training-knn.sh finalize [...]` – run the finalize stage locally.
- `sbatch training/training-knn.sh sweeps [...]` – launch individual worker
  tasks (used internally by the submission helper).

The XGBoost launcher exposes identical hooks (`plan`, `finalize`, `sweeps`) and
forwards every additional CLI flag to `python -m xgb.pipeline`. Hyper-parameters
such as `--feature-spaces`, `--learning-rate-grid`, or environment overrides
like `WORD2VEC_WORKERS` and `XGB_TREE_METHOD` behave exactly as they do when the
Python modules are called directly.

> **Note**: The sweep workers import Hugging Face `datasets` and
> `sentence-transformers`. Ensure the Python environment visible to `sbatch`
> already has `pip install -r development/requirements-dev.txt` (or at minimum
> `pip install datasets sentence-transformers`). The launchers perform a
> pre-flight import check and will abort with an actionable message if a
> dependency is missing.

By default the scripts reserve a single GPU (`--partition=gpu --gres=gpu:1 --nodes=1`),
16 CPU cores, and 128 GB of RAM for every sweep task (and the finalize stage).
Set `KNN_USE_GPU=0` / `XGB_USE_GPU=0` to keep everything on CPU, or narrow
`*_GPU_FEATURES` to a comma-separated subset (default `*` for all features).
You can also override `*_GPU_PARTITION`, `*_GPU_GRES`, `*_GPU_CPUS`, `*_GPU_MEM`,
`*_GPU_TIME`, and `*_GPU_SBATCH_FLAGS` to match your cluster’s scheduling
requirements.

Large grids are automatically split into chunks (default 1 000 tasks per
array). Tune this via `KNN_MAX_ARRAY_SIZE` / `XGB_MAX_ARRAY_SIZE` when the
cluster enforces a different limit.

See `recipes/README.md` for the configuration files consumed by the GRPO and
GRAIL launchers.

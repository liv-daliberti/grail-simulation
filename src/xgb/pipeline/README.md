# XGB Pipeline Package

`xgb.pipeline` orchestrates the multi-stage automation for the XGBoost slate
baseline: planning sweeps, executing runs, finalising winning models, and
regenerating reports. The package mirrors the structure used by the `knn`
pipeline for familiarity.

## Modules

- `cli.py` – high-level CLI invoked via `python -m xgb.pipeline`; exposes stage
  selection, job concurrency, cache reuse, and output directory flags.
- `__main__.py` – allows module execution with `python -m xgb.pipeline`.
- `context.py` – centralises filesystem paths, cache discovery, and dataset
  resolution for the pipeline stages.
- `evaluate.py` – final-stage orchestration that reloads winning configs, runs
  evaluations, and aggregates metrics.
- `__init__.py` – re-exports the primary entry points.

Stage-specific helpers live under subpackages:

- `sweeps/` – hyper-parameter grid definitions and execution utilities.
- `reports/` – Markdown builders for `reports/xgb`.

## Typical workflow

```bash
python -m xgb.pipeline --stage plan
python -m xgb.pipeline --stage sweeps --jobs 8
python -m xgb.pipeline --stage finalize --reuse-sweeps
python -m xgb.pipeline --stage reports --reuse-sweeps --reuse-final
```

The pipeline respects cached artefacts to avoid rerunning completed work.
Extend `context.py` and `evaluate.py` when new stages or artefacts are added.

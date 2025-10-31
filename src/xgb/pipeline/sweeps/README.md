# XGB Sweep Orchestration

`xgb.pipeline.sweeps` packages the logic that plans and executes hyper-parameter
searches for the XGBoost slate baseline. The helpers are consumed by
`xgb.pipeline` when running the `plan` and `sweeps` stages.

## Modules

- `common.py` – shared helpers for loading cached artefacts, scheduling CLI
  invocations, and recording trial outcomes.
- `next_video.py` – defines the hyper-parameter grid (learning rate, depth,
  estimators, text vectorizers) and execution flow for slate-accuracy training.
- `opinion.py` – analogous orchestration for opinion-shift sweeps; reuses the
  next-video configuration with opinion-specific metrics.
- `planning.py` – renders human-readable plan summaries consumed by the `plan`
  stage CLI.
- `__init__.py` – convenience exports for pipeline consumers.

When introducing new sweep dimensions or evaluation tasks, extend the grid
definitions here so the pipeline, reports, and automation inherit the change.

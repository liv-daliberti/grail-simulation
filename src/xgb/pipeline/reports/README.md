# XGB Pipeline Reports

This package renders Markdown reports for the XGBoost baseline’s pipeline
(`python -m xgb.pipeline --stage reports`). The modules transform cached sweep
and final-run artefacts into human-readable summaries under `reports/xgb/`.

## Modules

- `catalog.py` – top-level coordinator that loads cached metrics and hands them
  to the appropriate report builders.
- `features.py` – tables explaining feature-space performance and configuration
  details.
- `hyperparameter.py` – summaries of sweep grids, trial counts, and
  best-performing configurations.
- `next_video.py` – narrative plus tables/plots for slate-accuracy evaluation.
- `plots.py` – helpers to embed pre-rendered figures and generate Matplotlib
  charts on the fly.
- `runner.py` – command-line helper invoked by the pipeline; resolves paths and
  dispatches to the catalog.
- `shared.py` – formatting helpers reused across report sections.

Subpackage:

- `opinion/` – detailed opinion-shift report writers (see its README).

When adding new artefacts to the pipeline, extend `catalog.py` and add the
supporting builders so the generated Markdown stays in sync with the metrics.

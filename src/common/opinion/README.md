# Opinion Evaluation Helpers

`common.opinion` holds the shared utilities for modelling and analysing opinion
shifts. Baselines reuse these modules to ensure consistent metrics and report
generation across pipelines.

## Modules

- `metrics.py` – core metric calculations (MAE, RMSE, R², direction accuracy,
  calibration) plus helpers for aggregating across cohorts.
- `models.py` – lightweight model abstractions used by baselines when fitting
  opinion regressors.
- `plots.py` – Matplotlib helpers for calibration curves, scatter plots, and
  distribution comparisons.
- `prompts.py` – prompt construction helpers tailored to opinion tasks (e.g.,
  injecting pre/post survey context).
- `results.py` – dataclasses that capture evaluation outputs in a structured
  way for downstream reports.
- `sweep_types.py` – type definitions shared by the pipeline sweep planners.
- `__init__.py` – convenience exports.

Add new opinion metrics or plots here so every baseline inherits the change
without additional wiring.

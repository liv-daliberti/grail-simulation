# XGB Opinion Reports

`xgb.pipeline.reports.opinion` produces the opinion-shift sections of the
pipeline reports. These modules turn cached evaluation artefacts into tables,
plots, and narrative summaries under `reports/xgb/opinion/`.

## Modules

- `accumulators.py` – aggregation helpers that collect metrics across studies,
  feature spaces, and trial configurations.
- `curves.py` – utilities that format MAE/RMSE/R²-by-step curves for plots and
  table exports.
- `metrics.py` – transforms raw evaluation JSON into structured dataclasses used
  throughout the report builders.
- `observations.py` – generates call-outs and descriptive text highlighting key
  changes or notable results.
- `prediction_plots.py` – renders calibration and scatter plots that compare
  predicted vs. observed opinion shifts.
- `report.py` – top-level writer that stitches together all sections into a
  single Markdown document.
- `summaries.py` – concise summaries keyed by issue or participant cohort.
- `tables.py` – tabular views emphasising sample sizes, errors, and direction
  accuracy.

Extend the accumulators when new metrics are introduced so downstream builders
receive the additional context without duplication. All plot utilities are
designed to operate on cached NumPy/Pandas-friendly structures to keep the
report generation fast.

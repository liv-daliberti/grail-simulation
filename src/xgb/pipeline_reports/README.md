# Legacy XGB Pipeline Reports

`xgb.pipeline_reports` maintains backwards-compatible imports for older
tooling that still references the pre-refactor report modules. The code simply
re-exports the modern builders from `xgb.pipeline.reports`.

## Structure

- `catalog.py`, `features.py`, `hyperparameter.py`, `next_video.py`,
  `opinion.py`, `plots.py`, `runner.py`, `shared.py` – thin wrappers that
  import and forward calls to their counterparts in
  `xgb.pipeline.reports.*`.
- `__init__.py` – exposes the same surface as before so `from xgb.pipeline_reports import runner`
  continues to work.

When updating report behavior, make changes in `xgb.pipeline.reports` and keep
the shim modules importing from there. Avoid introducing new logic here—its
sole purpose is compatibility.

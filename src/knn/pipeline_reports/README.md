# Legacy KNN Pipeline Reports

`knn.pipeline_reports` preserves the pre-refactor import paths for report
generators. The modules simply forward to the modern builders in
`knn.pipeline.reports`.

## Structure

- `features.py`, `hyperparameter.py`, `next_video.py`, `opinion.py`, `shared.py`
  – wrapper modules that import and re-export the corresponding functions from
  `knn.pipeline.reports`.
- `__init__.py` – maintains the old public API for compatibility.

When updating report logic, make the change in `knn.pipeline.reports` and keep
these shims in sync by forwarding to the new implementations.

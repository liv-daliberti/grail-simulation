# Evaluation Utilities

`common.evaluation` contains helpers for computing and summarising evaluation
metrics shared by the baseline packages. These modules are largely focused on
slate-quality analysis.

## Modules

- `slate_eval.py` – core evaluation routines for slate recovery tasks; computes
  accuracy, coverage, and supporting diagnostics.
- `matrix_summary.py` – helpers for summarising sparse/dense confusion matrices
  and exporting them to reports.
- `utils.py` – miscellaneous helpers (e.g., metric aggregation, safe divisions)
  reused across evaluations.
- `__init__.py` – convenience exports for downstream modules.

Use these utilities when introducing new evaluation scripts so baseline metrics
remain consistent.

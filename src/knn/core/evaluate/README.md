# KNN Evaluation Helpers

`knn.core.evaluate` bundles the routines used to score kNN models and export
diagnostics. The helpers feed both the CLI evaluations and the pipeline
reports.

## Modules

- `dataset_eval.py` – prepares evaluation datasets, handles leave-one-study-out
  splits, and wires pass-through metadata.
- `indexes.py` – helpers for loading/persisting evaluation-ready indices.
- `metrics.py` – evaluation metrics (accuracy, slate coverage, elbow tracking)
  and aggregation utilities.
- `k_selection.py` – elbow detection and best-`k` selection logic.
- `curves.py` – prepares accuracy/MAE-by-k curves for plotting.
- `outputs.py` – writes JSON/CSV artefacts with evaluation results.
- `pipeline.py` – orchestrates the end-to-end evaluation flow.
- `provenance.py` – captures metadata about datasets, feature spaces, and model
  revisions for reproducibility.
- `utils.py` – miscellaneous helpers shared across the evaluation submodules.
- `__init__.py` – exports the primary entry points.

Extend this package when adding new diagnostics so the CLI and pipeline can
reuse the logic without duplication.

# XGBoost Report Catalog

This directory mirrors the assets emitted by `python -m xgb.pipeline` and
`training/training-xgb.sh`:

- `next_video/` – selected slate-ranking evaluation summaries.
- `opinion/` – post-study opinion regression metrics derived from the same hyper-parameters.
- `hyperparameter_tuning/` – notes from the learning-rate/depth sweeps that choose the current configuration.

The pipeline regenerates these Markdown files on every run so that dashboards in
the repository stay in sync with the latest experiment sweep.

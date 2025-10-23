# XGBoost Report Catalog

The Markdown artefacts in this directory are produced by `python -m xgb.pipeline` (or `training/training-xgb.sh`) and track the XGBoost baselines that accompany the simulation:

- `hyperparameter_tuning/README.md` – sweep grids, configuration deltas, and parameter frequency summaries.
- `next_video/README.md` – validation accuracy, coverage, and probability diagnostics for the slate-ranking task.
- `opinion/README.md` – post-study opinion regression metrics with MAE deltas versus the no-change baseline.

Raw metrics, model checkpoints, and intermediate artefacts referenced by these reports live beneath `models/xgb/…`.

## Refreshing Reports

```bash
PYTHONPATH=src python -m xgb.pipeline --stage full \
  --out-dir models/xgb \
  --reports-dir reports/xgb
```

Stages can be invoked individually (`plan`, `sweeps`, `finalize`, `reports`) to match existing SLURM workflows.

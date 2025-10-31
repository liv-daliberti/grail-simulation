# XGBoost Training Helpers

`common.ml.xgb` offers reusable utilities for training XGBoost models. The
helpers are consumed by `xgb.core` to keep the main modules focused on pipeline
logic.

## Modules

- `callbacks.py` – custom callback implementations (logging, early stopping,
  checkpoint reporting) tailored to the slate baseline.
- `fit_utils.py` – shared fit-time helpers (parameter preparation, booster
  configuration, evaluation-set wiring).
- `__init__.py` – exposes the public helpers.

Extend this package when new training conveniences are required, keeping the
logic generic enough to be reused by future XGBoost-based experiments.

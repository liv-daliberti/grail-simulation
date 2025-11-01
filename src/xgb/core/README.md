# XGB Core Components

`xgb.core` houses the reusable building blocks for the XGBoost slate baseline:
dataset loaders, feature builders, model wrappers, and evaluation utilities.
Both the CLI (`xgb.cli`) and pipeline (`xgb.pipeline`) import these modules.

## Modules

- `data.py` – dataset loading and filtering (studies, issues, splits) with
  optional caching hooks.
- `features.py` – prompt document assembly plus feature matrix construction.
- `vectorizers.py` – adapters for TF-IDF, Word2Vec, and sentence-transformer
  embeddings, including persistence helpers.
- `model.py` – wraps XGBoost training, checkpointing, and inference.
- `evaluate.py` – orchestration for next-video and opinion evaluations.
- `opinion.py` – opinion-specific evaluation helpers (metrics, dataset
  preparation).
- `utils.py` – shared helpers (filesystem, logging, configuration validation).
- `_optional.py` – lazy import helpers for optional heavy dependencies.
- `evaluation_metrics.py`, `evaluation_probabilities.py`,
  `evaluation_records.py`, `evaluation_types.py` – structured containers and
  helpers for recording evaluation outcomes in a consistent format.

Extend this package when introducing new feature spaces or evaluation modes so
the higher-level CLIs remain thin orchestration layers. Keep expensive imports
behind `_optional.py` to maintain fast startup times.

## Model Module Split (Refactor)

To improve maintainability and satisfy lint constraints, the original
`model.py` has been split into a few focused modules while keeping the public
API stable via re-exports:

- `model_types.py` – lightweight dataclasses (e.g. `XGBoostSlateModel`,
  `EncodedDataset`, `TrainingBatch`).
- `model_config.py` – configuration dataclasses for training and booster
  hyper-parameters (`XGBoostTrainConfig`, `XGBoostBoosterParams`). The
  constructors accept grouped configs and `**kwargs` for backwards
  compatibility with flat argument styles.
- `model_predict.py` – persistence (`save_xgboost_model`, `load_xgboost_model`)
  and inference (`predict_among_slate`) helpers.

All of the above are re-exported from `model.py`, so existing imports like
`from xgb.model import XGBoostTrainConfig, save_xgboost_model` continue to work
unchanged.

## Lightweight Imports for Tests

Some submodules depend on the KNN pipeline. For unit tests that don’t need the
full dependency surface, set the following environment variables before
importing `xgb` modules:

- `XGB_LIGHT_IMPORTS=1` – avoids initializing CLI/pipeline in `xgb.__init__`.
- `XGB_CORE_LIGHT_IMPORTS=1` – avoids importing `data`/`evaluate` in
  `xgb.core.__init__`.

With these enabled, importing `xgb.core.model` and the related split modules
does not pull in `knn` or other heavy dependencies.

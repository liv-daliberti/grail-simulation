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

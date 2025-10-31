# KNN Core Components

`knn.core` provides the shared building blocks for the kNN slate baseline:
dataset loaders, feature builders, index management, and opinion-evaluation
helpers. The CLI and pipeline layers import these modules directly.

## Modules

- `data.py` – dataset loading, filtering (studies, issues), and split handling.
- `features.py` – prompt document assembly plus feature matrix construction for
  TF-IDF, Word2Vec, and sentence-transformer spaces.
- `index.py` – index builders, persistence helpers, and search utilities for the
  different feature spaces.
- `opinion_*` modules – opinion-specific dataset adapters (`opinion_data.py`),
  index construction (`opinion_index.py`), model wrappers (`opinion_models.py`),
  output formatting (`opinion_outputs.py`), plot helpers (`opinion_plots.py`),
  and prediction export logic (`opinion_predictions.py`).
- `opinion.py` – orchestrates opinion evaluation using the helpers above.
- `utils.py` – shared utilities (logging, file system helpers).
- `evaluate/` – evaluation pipeline helpers (see its README).

Keep heavy dependencies optional and focus new shared logic here so both the CLI
and pipeline layers stay lean.

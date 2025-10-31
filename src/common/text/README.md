# Text Processing Utilities

`common.text` provides shared text-processing helpers: embedding loaders,
vectoriser wrappers, and indexing utilities. Baselines import these modules to
avoid duplicating feature engineering code.

## Modules

- `embeddings.py` – sentence-transformer configuration (model selection,
  batching, device placement) plus pooling/normalisation helpers.
- `vectorizers.py` – TF-IDF and Word2Vec wrappers that expose consistent fit /
  transform APIs and persistence helpers.
- `title_index.py` – builds lookup tables for video titles so evaluations can
  resolve IDs back to readable text.
- `utils.py` – miscellaneous text helpers (tokenisation, normalisation,
  stopword handling).
- `__init__.py` – exports the main helpers for consumers.

Keep heavy dependencies optional—callers should guard imports so tests without
GPU/large models remain lightweight.

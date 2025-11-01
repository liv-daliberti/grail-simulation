# KNN CLI Package

`knn.cli` provides the command-line interface for the k-nearest-neighbor slate
baseline. It exposes a single entry point (`python -m knn.cli`) that can train
indices, evaluate runs, and export artifacts.

## Modules

- `main.py` – argument parser and runtime driver; wires feature-space options,
  dataset filtering flags, and evaluation controls.
- `utils.py` – shared argparse helpers (sentence-transformer flag groups, common
  path validation).
- `__main__.py` – enables execution via `python -m knn.cli`.
- `__init__.py` – re-exports the CLI helpers.

Common usage:

```bash
export PYTHONPATH=src
python -m knn.cli \
  --dataset data/cleaned_grail \
  --out_dir models/knn/example \
  --feature_space tfidf \
  --fit_index
```

Run `python -m knn.cli --help` to inspect all available switches.

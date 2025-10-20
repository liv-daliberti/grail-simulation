# KNN Slate Baselines (Refactor in Progress)

The legacy `knn-baseline.py` script is being migrated into a modular
package so we can experiment with richer feature spaces and cleaner
evaluation tooling. The refactor introduces a package layout and Word2Vec
embedding pipeline that will replace the single-file implementation.

## Package layout

```
src/knn/
├── __init__.py        # package entry point
├── cli.py             # argument parsing / orchestration (upcoming)
├── data.py            # dataset loading, filtering, slate bucketing (stub)
├── evaluate.py        # metrics for predicted vs. gold slates (stub)
├── features.py        # TF-IDF + Word2Vec feature builders (stub + W2V skeleton)
├── index.py           # reusable KNN index wrapper (stub)
├── utils.py           # logging / filesystem helpers
└── knn-baseline.py    # legacy executable script (current baseline)
```

The new modules are currently scaffolds with docstrings and
`NotImplementedError` placeholders. Functionality will move into them in
follow-up PRs while keeping the legacy script working for reproducibility.

## Word2Vec migration

`features.py` now contains a `Word2VecFeatureBuilder` skeleton that will:

1. Build prompt text via `prompt_builder.build_user_prompt` to ensure
   the same PROFILE/HISTORY context used elsewhere in the project.
2. Tokenise the prompt text and train a gensim Word2Vec model with
   configurable vector size, window, min-count, and epochs.
3. Persist models under `models/knn_word2vec/` so embeddings can be
   reused between runs.
4. Produce averaged word-vector embeddings for each slate as the new
   feature space, while retaining a TF-IDF fallback for ablation.

The upcoming CLI (`cli.py`) will expose knobs such as `--feature-space
tfidf|word2vec`, `--vector-size`, `--window`, and `--min-count`, and will
delegate to the appropriate feature/index modules.

## Migration checklist

- [ ] Port dataset loading, slate bucketing, and filtering from
      `knn-baseline.py` into `data.py`.
- [ ] Move TF-IDF vectoriser code into `features.py` and add unit tests.
- [ ] Implement `KNNIndex` (fit/save/load/predict) in `index.py`.
- [ ] Implement evaluation metrics (accuracy, option coverage) in
      `evaluate.py`.
- [ ] Wire the CLI to orchestrate load → feature → index → evaluate,
      using the new Word2Vec defaults.
- [ ] Add tests under `tests/knn/` covering feature extraction and index
      predictions.

Until the refactor is complete, continue using `knn-baseline.py` directly
for experiments. Contributions that flesh out the new modules are very
welcome—just make sure to keep backwards compatibility paths during the
transition.

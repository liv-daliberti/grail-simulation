# XGBoost Slate Baselines

This package mirrors the structure of the refactored `knn` baseline while
swapping the neighbourhood search for an XGBoost multi-class classifier. The
goal is to provide a fast, non-generative baseline that can be compared against
the kNN approach and any future neural models.

## Package layout

```
src/xgboost/
├── __init__.py          # package entry point
├── cli.py               # argument parsing / orchestration
├── data.py              # dataset loading helpers (re-exported from knn)
├── evaluate.py          # evaluation loop + metrics
├── features.py          # prompt assembly utilities shared with knn
├── model.py             # TF-IDF + XGBoost training/prediction helpers
├── utils.py             # logging / filesystem helpers
└── xgboost-baseline.py  # legacy executable entry point
```

The modules intentionally mirror the evolving `knn` refactor so experiments can
share most of the prompt-building and dataset plumbing. Feature construction is
currently TF-IDF based; switching to more sophisticated embeddings should only
require touching `model.py`.

## Quick start

Train and evaluate on all available issues:

```bash
python -m xgboost.cli \
  --fit_model \
  --dataset data/cleaned_grail \
  --out_dir models/xgboost/run-001
```

To speed up experimentation you can pre-train per-issue models and reuse them:

```bash
python -m xgboost.cli \
  --fit_model \
  --save_model models/xgboost/checkpoints

python -m xgboost.cli \
  --load_model models/xgboost/checkpoints \
  --out_dir models/xgboost/eval \
  --issues minimum_wage,gun_control
```

## Implementation notes

* TF-IDF features are generated via the shared prompt builder to stay aligned
  with other baselines.
* XGBoost is configured with `multi:softprob`, allowing the model to emit a
  probability for every seen video id; slate candidates are re-ranked using the
  highest available probability.
* Metrics include overall accuracy plus a "coverage" score indicating how often
  the correct option was among the candidate ids observed during training.

Pull requests extending the feature pipeline or adding richer metrics are very
welcome. Keep the CLI backwards compatible where possible so scripts that
exercise the baseline do not need to change.

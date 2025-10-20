# KNN Slate Baselines

Modular implementation of the k-nearest-neighbour slate selector. The package
supersedes the legacy `knn-baseline.py` script while keeping the single-file
entry point for backwards compatibility.

## Package layout

```
src/knn/
├── cli.py          # CLI front-end (train/evaluate, issue filtering, exports)
├── data.py         # dataset loading helpers + issue-aware filtering
├── evaluate.py     # accuracy + coverage metrics and evaluation loop
├── features.py     # TF-IDF + optional Word2Vec document builders
├── index.py        # Faiss / sklearn KNN wrapper with persistence helpers
├── utils.py        # logging, prompt helpers, and video-id canonicalisation
└── knn-baseline.py # legacy entry point delegating to knn.cli:main
```

## Quick start

Train an index and evaluate on the default cleaned dataset:

```bash
python -m knn.cli \
  --dataset data/cleaned_grail \
  --out-dir models/knn/run-001 \
  --feature-space tfidf \
  --fit-index
```

To compare against Word2Vec features and a filtered issue subset:

```bash
python -m knn.cli \
  --dataset data/cleaned_grail \
  --out-dir models/knn/run-w2v \
  --feature-space word2vec \
  --issue minimum_wage \
  --word2vec-size 256 \
  --eval-max 2000
```

All CLI switches are documented via `python -m knn.cli --help`. The script writes
predictions, metrics, and optional embeddings under the specified `out_dir`.

## Feature helpers

`features.py` reuses `prompt_builder.build_user_prompt` to guarantee the same
PROFILE/HISTORY context seen by other baselines:

- TF-IDF is enabled by default, with optional extra context using
  `--knn-text-fields`.
- Word2Vec training (via gensim) can be toggled with `--feature-space word2vec`;
  models persist to `models/knn_word2vec/<issue>/` by default so they can be reused.
- Title lookups pull from metadata CSVs listed by `GRAIL_TITLE_*` environment
  variables, falling back to the shared network drive defaults.

## Evaluation

`evaluate.py` computes accuracy and slate coverage—the latter surfaces how often
the gold video appears in the candidate slate. Metrics mirror those reported by
the XGBoost and GPT-4o baselines so results remain comparable.

## Testing

Unit tests live under `tests/knn/`. Add or update fixtures when introducing new
feature builders or storage formats. The CLI smoke tests rely on small synthetic
datasets generated during CI to avoid pulling the full cleaned corpus.

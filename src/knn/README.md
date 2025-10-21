# KNN Slate Baselines

Modular implementation of the k-nearest-neighbor slate selector. The package
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
├── utils.py        # logging, prompt helpers, and video-id canonicalization
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

## Pipeline overview

1. **Feature extraction** – `src/knn/features.py` assembles per-issue text documents and can train either TF-IDF vectors or Word2Vec embeddings via `Word2VecFeatureBuilder`.
2. **Index training** – `src/knn/index.py` fits the requested space (`build_tfidf_index` / `build_word2vec_index`) and persists artifacts so later runs can reuse them.
3. **Evaluation & elbow selection** – `src/knn/evaluate.py` scores validation examples, logs running accuracies, generates accuracy-by-`k` curves, and picks the elbow-derived `k`.
4. **Reporting** – evaluation metrics, per-`k` predictions, elbow plots, and curve diagnostics write to `models/` and `reports/`.

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

Each run also materializes elbow plots and curve summaries:

- Elbow charts are saved to `reports/knn/<feature-space>/elbow_<issue>.png`.
- Per-`k` predictions and metrics live under `models/knn/<issue>/k-<k>/`.
- Curve diagnostics (accuracy-by-k, AUC, best-k) for both evaluation and training
  splits are written to `models/knn/<issue>/knn_curves_<issue>.json`. Use
  `--train-curve-max` to cap the number of training examples analyzed.

## Hyperparameter sweeps

We keep curated sweeps for both feature spaces under `models/knn/sweeps/` (see
`reports/knn/hyperparameter_tuning.md` for the latest summary). Each configuration
evaluates

- `k ∈ {1,2,3,4,5,10,15,20,25,50,75,100}`
- distance metrics `cosine` and `l2`
- optional text augmentation (`viewer_profile,state_text`)
- Word2Vec dimensions (`128`, `256`) and windows (`5`, `10`)

Example TF-IDF sweep:

```bash
export PYTHONPATH=src
for issue in minimum_wage gun_control; do
  for metric in cosine l2; do
    for fields in "" "viewer_profile,state_text"; do
      label=${fields:-none}
      python -m knn.cli \
        --dataset data/cleaned_grail \
        --fit-index \
        --feature-space tfidf \
        --issues "$issue" \
        --knn_k 25 \
        --knn_k_sweep 1,2,3,4,5,10,15,20,25,50,75,100 \
        --knn_metric "$metric" \
        --knn_max_train 5000 \
        --eval_max 200 \
        --train_curve_max 2000 \
        --cache_dir hf_cache \
        --out_dir "models/knn/sweeps/tfidf/${issue}/metric-${metric}_text-${label}" \
        $( [ -n "$fields" ] && printf -- '--knn_text_fields %s' "$fields" )
    done
  done
done
```

and the corresponding Word2Vec sweep:

```bash
export PYTHONPATH=src WORD2VEC_WORKERS=40
for issue in minimum_wage gun_control; do
  for metric in cosine l2; do
    for fields in "" "viewer_profile,state_text"; do
      label=${fields:-none}
      for size in 128 256; do
        for window in 5 10; do
          python -m knn.cli \
            --dataset data/cleaned_grail \
            --fit-index \
            --feature-space word2vec \
            --issues "$issue" \
            --knn_k 25 \
            --knn_k_sweep 1,2,3,4,5,10,15,20,25,50,75,100 \
            --knn_metric "$metric" \
            --knn_max_train 5000 \
            --eval_max 200 \
            --train_curve_max 2000 \
            --cache_dir hf_cache \
            --word2vec-model-dir models/knn_word2vec_sweeps \
            --word2vec-size "$size" \
            --word2vec-window "$window" \
            --word2vec-min-count 1 \
            --word2vec-epochs 10 \
            --word2vec-workers "${WORD2VEC_WORKERS:-40}" \
            --out_dir "models/knn/sweeps/word2vec/${issue}/metric-${metric}_text-${label}_sz${size}_win${window}_min1" \
            $( [ -n "$fields" ] && printf -- '--knn_text_fields %s' "$fields" )
        done
      done
    done
  done
done
```

The loops mirror the runs referenced in the report; feel free to expand the grid
with additional parameters.

## Testing

Unit tests live under `tests/knn/`. Add or update fixtures when introducing new
feature builders or storage formats. The CLI smoke tests rely on small synthetic
datasets generated during CI to avoid pulling the full cleaned corpus.

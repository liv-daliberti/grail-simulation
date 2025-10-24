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

High-level progression (training + evaluation):

```
        Raw Capsule Exports
                  |
                  v
        clean_data.sessions
                  |
                  v
         Prompt Builder (GRPO)
                  |
                  v
        +-----------------------+
        |  Feature Extraction   |
        |  - TF-IDF (default)   |
        |  - Word2Vec (optional)|
        +-----------+-----------+
                    |
            +-------v-------+
            |  KNN Index   |
            |  Training    |
            +-------+------+
                    |
      +-------------v--------------+
      | KNN Evaluation (metrics,   |
      | elbow curve, acc@k logs)   |
      +-------------+--------------+
                    |
        +-----------+-----------+
        | Reports & Artifacts  |
        |  models/, reports/   |
        +----------------------+
```

Both the SLURM wrapper (`training/training-knn.sh`, which auto-submits sweeps/finalize jobs) and the Python modules follow this path; setting `--feature-space word2vec` switches the feature block while keeping the rest intact.

Need deeper context on the ingestion step? See `clean_data/sessions/README.md` for the session-log builders that feed the dataset.

## Report generation modules

The report builder that previously lived entirely in `src/knn/pipeline_reports.py` now ships as the
package `src/knn/pipeline_reports/`. The public entry point remains
`knn.pipeline_reports.generate_reports(repo_root, report_bundle)` so existing imports and pipeline
calls continue to work, but the implementation is split into focused modules:

- `__init__.py` – orchestrates the overall workflow and wires the helper modules together.
- `catalog.py` – creates the top-level `reports/knn/README.md` summary.
- `hyperparameter.py` – renders sweep leaderboards, per-feature tables, and reproduction commands.
- `next_video.py` – assembles next-video metrics tables, curve plots, and LOSO summaries.
- `opinion.py` – produces opinion-regression tables, portfolio stats, and cross-study diagnostics.
- `shared.py` – reusable helpers such as CLI formatting, feature-space headings, and logging.

When you need to tweak layout or add new sections, update the dedicated module instead of hunting
through a monolithic file. Each builder writes a single Markdown artifact and accepts the same
`ReportBundle` produced by the pipeline, so the CLI and training scripts require no changes.

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

- Elbow charts are saved to `reports/knn/next_video/<feature-space>/elbow_<issue>.png`.
- Per-`k` predictions and metrics live under `models/knn/next_video/<feature-space>/<study>/`.
- Curve diagnostics (accuracy-by-k, AUC, best-k) for both evaluation and training
  splits are written to `models/knn/next_video/<feature-space>/<study>/<issue>/knn_curves_<issue>.json`.
  Use `--train-curve-max` to cap the number of training examples analyzed.

## Hyperparameter sweeps

We keep curated next-video sweeps under `models/knn/next_video/sweeps/` and opinion sweeps
under `models/knn/opinions/sweeps/` (see
`reports/knn/hyperparameter_tuning/README.md` for the latest summary). Each configuration
evaluates

- `k ∈ {1,2,3,4,5,10,15,20,25,50,75,100}`
- distance metrics `cosine` and `l2`
- default text augmentation combines the prompt builder document with `viewer_profile`
  and `state_text` (pass `--knn_text_fields` to append more columns)
- Word2Vec dimensions (`128`, `256`) and windows (`5`, `10`)

Example TF-IDF sweep:

```bash
export PYTHONPATH=src
for issue in minimum_wage gun_control; do
  for metric in cosine l2; do
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
      --out_dir "models/knn/next_video/sweeps/tfidf/${issue}/metric-${metric}_text-default"
  done
done
```
Include `--knn_text_fields field_a,field_b` in the command above to append additional
columns beyond the default viewer/profile context.

and the corresponding Word2Vec sweep:

```bash
export PYTHONPATH=src WORD2VEC_WORKERS=40
for issue in minimum_wage gun_control; do
  for metric in cosine l2; do
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
          --out_dir "models/knn/next_video/sweeps/word2vec/${issue}/metric-${metric}_text-default_sz${size}_win${window}_min1"
      done
    done
  done
done
```
As with the TF-IDF example, pass `--knn_text_fields` to augment the prompt document
beyond the default viewer profile fields.

The loops mirror the runs referenced in the report; feel free to expand the grid
with additional parameters.

## Testing

Unit tests live under `tests/knn/`. Add or update fixtures when introducing new
feature builders or storage formats. The CLI smoke tests rely on small synthetic
datasets generated during CI to avoid pulling the full cleaned corpus.

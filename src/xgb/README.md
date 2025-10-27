# XGBoost Slate Baseline

Tree-based counterpart to the `knn` package. This module trains gradient-boosted
classifiers over the same prompt documents, supports prompt embeddings beyond
TF-IDF, and ships with automated sweeps + report generation.

## Code Map

```
src/xgb/
├── cli.py / xgboost-baseline.py   # Single-run CLI (train, evaluate, export)
├── data.py                        # Dataset loading + issue/study filtering
├── evaluate.py                    # Training loop, metrics writer, CLI orchestration
├── features.py                    # Prompt assembly + slate extraction utilities
├── model.py                       # TF-IDF vectoriser + XGBoost wrappers
├── vectorizers.py                 # Word2Vec + SentenceTransformer adapters
├── opinion.py                     # Opinion-regression training/evaluation
├── pipeline.py                    # End-to-end sweeps, finals, and reports
├── pipeline_cli.py                # Pipeline argument parsing + default directories
├── pipeline_context.py            # Execution context (datasets, reports, sweeps)
├── pipeline_evaluate.py           # Final evaluation orchestration
├── pipeline_reports/              # Markdown builders for reports/xgb/*
├── pipeline_sweeps.py             # Hyper-parameter grids + job fan-out
├── pipeline_utils.py              # Shared helpers and cache management
└── utils.py                       # Logging, canonicalisation, prompt helpers
```

Set `PYTHONPATH=src` (or install the package) before invoking the CLIs.

## Quick Start (Single Run)

Fit a model and evaluate on the cleaned dataset:

```bash
export PYTHONPATH=src
python -m xgb.cli \
  --fit_model \
  --dataset data/cleaned_grail \
  --out_dir models/xgb/example-run
```

Persist the trained bundle for reuse and evaluate on a subset of studies:

```bash
python -m xgb.cli \
  --fit_model \
  --save_model models/xgb/checkpoints

python -m xgb.cli \
  --load_model models/xgb/checkpoints \
  --out_dir models/xgb/eval \
  --participant-studies study1,study2
```

Swap in alternative prompt embeddings via `--text_vectorizer`:

```bash
# Word2Vec features (gensim required)
python -m xgb.cli \
  --fit_model \
  --text_vectorizer word2vec \
  --word2vec_size 256 \
  --word2vec_model_dir models/xgb/word2vec_models \
  --out_dir models/xgb/word2vec-run

# Sentence-transformer embeddings (GPU optional)
python -m xgb.cli \
  --fit_model \
  --text_vectorizer sentence_transformer \
  --sentence_transformer_model sentence-transformers/all-mpnet-base-v2 \
  --sentence_transformer_device cuda \
  --sentence_transformer_batch_size 64 \
  --out_dir models/xgb/st-run
```

Run `python -m xgb.cli --help` to inspect all arguments (including `--extra_text_fields`
for prompt augmentation and the `--xgb_*` hyper-parameters).

## Pipeline Automation

`python -m xgb.pipeline` mirrors the multi-stage flow used for kNN:

1. Hyper-parameter sweeps across learning rate, depth, estimators, regularisers,
   and text vectorisers.
2. Selection of the best slate configuration per study followed by final
   evaluations (with optional checkpoint export).
3. Opinion-regression sweeps and evaluations that reuse the winning slate
   settings.
4. Markdown regeneration beneath `reports/xgb/`.

Typical invocations:

```bash
# Dry-run to inspect the execution plan
python -m xgb.pipeline --dry-run

# Launch sweeps only with 8 parallel jobs
python -m xgb.pipeline --stage sweeps --jobs 8

# Rebuild reports from cached sweeps/finals
python -m xgb.pipeline --stage reports --reuse-sweeps --reuse-final

# Evaluate a reduced grid on sentence-transformer features
python -m xgb.pipeline \
  --text-vectorizer-grid sentence_transformer \
  --sentence_transformer_model sentence-transformers/all-mpnet-base-v2 \
  --issues minimum_wage \
  --studies study1
```

Use `--tasks next_video,opinion` to control stages, `--learning-rate-grid` and
friends to trim sweeps, and `--no-reuse-sweeps` / `--no-reuse-final` to force
recomputation. The SLURM launcher `training/training-xgb.sh` orchestrates these
stages in production.

## Feature Spaces & Vectorisers

- **TF-IDF** (default) – fast, memory-light baseline built via `model.TfidfVectorizerBundle`.
- **Word2Vec** – relies on `vectorizers.Word2VecBundle`; persisted models live
  under the directory supplied via `--word2vec_model_dir`.
- **Sentence Transformer** – handled by `vectorizers.SentenceTransformerBundle`.
  Normalisation flags are shared with the CLI via `common.cli_args`.

All vectorisers reuse `features.prepare_prompt_documents` to keep prompts aligned
with the other baselines.

## Opinion Workflow & Reports

`opinion.py` reuses the selected slate configuration to train regression models
per opinion study. Artefacts flow to `models/xgb/opinion/*`, while
`pipeline_reports/` produces Markdown under `reports/xgb/`:

- `catalog.py` builds the README summary.
- `hyperparameter.py` publishes sweep tables and best-config recaps.
- `opinion.py` renders opinion leadership tables and diagnostics.
- `shared.py` houses layout helpers used across the report modules.

Regenerate reports with `python -m xgb.pipeline --stage reports`.

## Outputs & Testing

- Slate metrics, predictions, and curve snapshots land in
  `models/xgb/next_video/<vectorizer>/<study>/<issue>/`.
- Saved model bundles are serialised via `model.save_model_bundle`.
- Opinion metrics mirror the layout under `models/xgb/opinion/<study>/`.

Tests live in `tests/xgb/` and cover CLI glue, sweep planning, and vectoriser
behaviour. Extend them when adding new feature spaces or pipeline stages so CI
remains deterministic without access to the full dataset.

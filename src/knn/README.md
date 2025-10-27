# KNN Slate Baselines

Modernised KNN slate selector with reusable feature builders, index helpers, and
pipeline automation. The package replaces the one-off `knn-baseline.py` script
while keeping a compatible CLI for ad-hoc experiments and batch jobs.

## Code Map

```
src/knn/
├── cli.py / knn-baseline.py       # Single-run CLI (train, evaluate, export artefacts)
├── cli_utils.py                   # Shared argparse helpers (e.g. ST embedding flags)
├── data.py                        # Dataset loading + issue/study filtering
├── features.py                    # Prompt document assembly + TF-IDF/Word2Vec glue
├── index.py                       # Index builders + persistence for TF-IDF/W2V/ST
├── evaluate.py                    # Training/eval loop, metrics, elbow selection
├── opinion.py / opinion_sweeps.py # Opinion-regression fit/eval + sweep definitions
├── pipeline.py                    # End-to-end sweeps, finals, and report generation
├── pipeline_cli.py                # Top-level pipeline CLI + default paths
├── pipeline_context.py            # Path resolution + execution configuration
├── pipeline_data.py               # Study/issue metadata helpers
├── pipeline_evaluate.py           # Cross-study + opinion evaluation orchestration
├── pipeline_io.py                 # Sweep/final metrics loading + JSON writers
├── pipeline_reports/              # Markdown builders for reports/knn/*
├── pipeline_sweeps.py             # Hyper-parameter grids and task partitioning
├── pipeline_utils.py              # Fan-out helpers shared across stages
└── utils.py                       # Logging, canonicalisation, slate utilities
```

Set `PYTHONPATH=src` (or install the package) before invoking any module-level CLI.

## Quick Start (Single Run)

Train and evaluate a TF-IDF index on the default cleaned dataset:

```bash
export PYTHONPATH=src
python -m knn.cli \
  --dataset data/cleaned_grail \
  --out-dir models/knn/example-run \
  --feature-space tfidf \
  --fit-index
```

Switch the feature space, limit issues, or reuse a persisted index:

```bash
python -m knn.cli \
  --dataset data/cleaned_grail \
  --out-dir models/knn/word2vec-demo \
  --feature-space word2vec \
  --issues minimum_wage,gun_control \
  --word2vec-size 256 \
  --fit-index \
  --save-index models/knn/word2vec-demo/index_cache

python -m knn.cli \
  --dataset data/cleaned_grail \
  --load-index models/knn/word2vec-demo/index_cache \
  --feature-space word2vec \
  --eval-max 2000 \
  --participant-studies study1,study2
```

All switches are documented via `python -m knn.cli --help`, including the
`sentence_transformer` flags for GPU-backed embeddings.

## Pipeline Automation

`python -m knn.pipeline` drives the full workflow used in SLURM jobs and CI:

1. Hyper-parameter sweeps for next-video ranking across requested feature spaces
   (`tfidf`, `word2vec`, `sentence_transformer`).
2. Final evaluations that reload the winning configuration per study and export
   metrics, predictions, and elbow curves.
3. Optional opinion-regression sweeps that reuse the slate configs.
4. Markdown regeneration under `reports/knn/`.

Common invocations:

```bash
# Inspect the plan without scheduling jobs
python -m knn.pipeline --dry-run

# Run the sweeps stage only (results land under models/knn/next_video/sweeps)
python -m knn.pipeline --stage sweeps --jobs 8

# Regenerate reports using cached sweeps/finals
python -m knn.pipeline --stage reports --reuse-sweeps --reuse-final

# Restrict to opinion tasks and sentence-transformer features
python -m knn.pipeline \
  --tasks opinion \
  --feature-spaces sentence_transformer \
  --sentence-transformer-model sentence-transformers/all-mpnet-base-v2
```

Use `--no-reuse-sweeps` or `--no-reuse-final` to force reruns. The SLURM wrapper
`training/training-knn.sh` simply orchestrates these stages on the cluster.

## Feature Spaces

- **TF-IDF (default)** – `features.prepare_prompt_documents` builds consistent
  prompt text; vectors are stored alongside trained `sklearn` indexes.
- **Word2Vec** – optional dependency on `gensim`. Models persist to
  `models/knn/next_video/word2vec_models/<issue>/<study>` so future runs can skip
  retraining.
- **Sentence Transformer** – uses `sentence-transformers` to encode prompts.
  Configure via `--sentence-transformer-*` flags in both the single-run CLI and
  pipeline. Normalisation can be toggled with `--sentence-transformer-normalize`.

All spaces share the prompt builder from `common.prompt_docs` to guarantee parity
with the GPT-4o and XGBoost baselines.

## Evaluation Outputs

`evaluate.py` computes accuracy and slate coverage plus elbow diagnostics. Expect:

- Metrics JSON/CSV under `models/knn/next_video/<feature>/<study>/<issue>/`.
- Per-`k` curves (`knn_curves_<issue>.json`) and elbow PNGs in the same folder.
- Opinion pipelines emit regression summaries beneath `models/knn/opinion/*` and
  populate `reports/knn/opinion/*.md` via the builders in `pipeline_reports/`.

`pipeline_reports/catalog.py` regenerates the top-level README, while the
issue-specific modules handle table/plot assembly.

## Testing

Targeted tests live under `tests/knn/`:

- `test_pipeline_modules.py` exercises sweep planning and CLI wiring.
- `test_sentence_transformer_index.py` validates sentence-transformer caching.
- Additional fixtures cover opinion regressions and dataset adapters.

Extend these when introducing new feature spaces or storage formats so CI can run
without downloading the full cleaned corpus.

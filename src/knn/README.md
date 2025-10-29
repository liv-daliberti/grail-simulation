# KNN Slate Baselines

Modernised KNN slate selector with reusable feature builders, index helpers, and
pipeline automation. The package replaces the one-off `knn-baseline.py` script
while keeping a compatible CLI for ad-hoc experiments and batch jobs.

## Code Map

```
src/knn/
├── cli/                           # Single-run CLI entry points
│   ├── __init__.py                # Public exports for python -m knn.cli
│   ├── main.py                    # Argument parsing, CLI wiring, eval dispatch
│   └── utils.py                   # Shared argparse helpers (e.g. ST flags)
├── core/                          # Feature extraction, indexing, evaluation
│   ├── __init__.py
│   ├── data.py                    # Dataset loading + issue/study filtering
│   ├── evaluate.py                # Training/eval loop, metrics, elbow selection
│   ├── features.py                # Prompt document assembly + TF-IDF/W2V glue
│   ├── index.py                   # Index builders + persistence for TF-IDF/W2V/ST
│   ├── opinion.py                 # Opinion-regression fit/eval helpers
│   └── utils.py                   # Logging + filesystem shims
├── pipeline/                      # End-to-end sweeps, finals, report generation
│   ├── __init__.py                # Orchestration entry point (python -m knn.pipeline)
│   ├── cli.py                     # Top-level pipeline CLI + default paths
│   ├── context.py                 # Path resolution + execution configuration
│   ├── data.py                    # Study/issue metadata helpers
│   ├── evaluate.py                # Final + opinion evaluation orchestration
│   ├── io.py                      # Sweep/final metrics loading + JSON writers
│   ├── sweeps.py                  # Hyper-parameter grids and task partitioning
│   ├── opinion_sweeps.py          # Opinion-specific sweep helpers
│   ├── utils.py                   # Fan-out helpers shared across stages
│   └── reports/                   # Markdown builders for reports/knn/*
└── scripts/
    ├── __init__.py
    └── baseline.py                # Backwards-compatible shim for knn-baseline.py
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
3. Opinion-regression sweeps that reuse the slate configs (runs by default; use
   `--tasks` to restrict to a subset, e.g. `--tasks next_video`).
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

### Defaults

The training launcher applies sensible defaults that mirror typical runs:

- Tasks: `next_video,opinion` (set via `--tasks` or `KNN_PIPELINE_TASKS`).
- Feature spaces: `tfidf,word2vec,sentence_transformer` (env `KNN_FEATURE_SPACES`).
- K sweep: `1,2,3,4,5,10,25,50` (env `KNN_K_SWEEP`).
- K selection: `elbow` (env `KNN_K_SELECT_METHOD`).
- Prompt text variants: first 5 merged extra-field options plus one aggregate covering all 15 (env `KNN_*_TEXT_LIMIT`).
- Distance metrics: TF-IDF sweeps cosine + L2; Word2Vec and Sentence Transformer default to cosine (env `KNN_*_METRICS`).
- Sentence-Transformer device: `cuda` when GPUs are enabled; override with
  `SENTENCE_TRANSFORMER_DEVICE` or `KNN_SENTENCE_DEVICE`.
- GPU scheduling is enabled by default in the launcher (`KNN_USE_GPU=1`); set
  `KNN_USE_GPU=0` to force CPU-only execution.

## Feature Spaces

- **TF-IDF (default)** – `features.prepare_prompt_documents` builds consistent
  prompt text; vectors are stored alongside trained `sklearn` indexes.
- **Word2Vec** – optional dependency on `gensim`. Models persist to
  `models/knn/next_video/word2vec_models/<issue>/<study>` so future runs can skip
  retraining.
- **Sentence Transformer** – uses `sentence-transformers` to encode prompts.
  Configure via `--sentence-transformer-*` flags in both the single-run CLI and
  pipeline. Normalisation can be toggled with `--sentence-transformer-normalize`.

All spaces share the prompt builder from `common.prompts.docs` to guarantee parity
with the GPT-4o and XGBoost baselines.

## Evaluation Outputs

`core/evaluate.py` computes accuracy and slate coverage plus elbow diagnostics. Expect:

- Metrics JSON/CSV under `models/knn/next_video/<feature>/<study>/<issue>/`.
- Per-`k` curves (`knn_curves_<issue>.json`) and elbow PNGs in the same folder.
- Opinion pipelines emit regression summaries beneath `models/knn/opinion/*` and
  populate `reports/knn/opinion/*.md` via the builders in `pipeline/reports/`.
- Opinion-from-next evaluations reuse the selected next-video configuration to score
  opinion change; metrics live under `models/knn/opinions/from_next/*` and drive the
  optional `reports/knn/opinion_from_next/` section.

`pipeline/reports/catalog.py` regenerates the top-level README, while the
issue-specific modules handle table/plot assembly.

## Testing

Targeted tests live under `tests/knn/`:

- `test_pipeline_modules.py` exercises sweep planning and CLI wiring.
- `test_sentence_transformer_index.py` validates sentence-transformer caching.
- Additional fixtures cover opinion regressions and dataset adapters.

Extend these when introducing new feature spaces or storage formats so CI can run
without downloading the full cleaned corpus.

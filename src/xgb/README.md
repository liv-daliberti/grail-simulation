# XGBoost Slate Baseline

Tree-based counterpart to the `knn` package. This module trains gradient-boosted
classifiers over the same prompt documents, supports prompt embeddings beyond
TF-IDF, and ships with automated sweeps + report generation.

## Code Map

```
src/xgb/
├── cli/                         # Single-run CLI (train, evaluate, export)
│   ├── __init__.py              # Package exports for scripting
│   ├── __main__.py              # Enables `python -m xgb.cli`
│   └── main.py                  # Training/evaluation argument parser
├── core/                        # Dataset loading, feature prep, models, evaluation
│   ├── __init__.py
│   ├── data.py                  # Dataset loading + issue/study filtering
│   ├── features.py              # Prompt document assembly helpers
│   ├── vectorizers.py           # TF-IDF/Word2Vec/SentenceTransformer wrappers
│   ├── model.py                 # Booster training + persistence helpers
│   ├── opinion.py               # Opinion-regression fit/eval utilities
│   └── evaluate.py              # Next-video + opinion evaluation orchestration
├── pipeline/                    # Hyper-parameter sweeps, finals, report generation
│   ├── __init__.py              # Orchestration entry point (python -m xgb.pipeline)
│   ├── __main__.py              # Enables `python -m xgb.pipeline`
│   ├── cli.py                   # Top-level pipeline CLI + default paths
│   ├── context.py               # Path resolution + execution configuration
│   ├── evaluate.py              # Finalization + opinion execution
│   ├── sweeps.py                # Hyper-parameter grids and task partitioning
│   └── reports/                 # Markdown builders for reports/xgb/*
├── pipeline_reports/            # Legacy shims forwarding to pipeline/reports/*
├── scripts/                     # Backwards-compatible entry points
└── xgboost-baseline.py          # Legacy CLI shim (invokes xgb.cli.main)
```

Existing imports that target `xgb.pipeline_reports` continue to work through the compatibility
wrappers; new code should import from `xgb.pipeline.reports`.

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
for prompt augmentation and the `--xgb_*` hyper-parameters). Legacy import paths such
as `xgb.data` and `xgb.pipeline_cli` remain available via compatibility aliases.

## Pipeline Automation

`python -m xgb.pipeline` mirrors the multi-stage flow used for kNN:

1. `plan` stage enumerates sweep tasks, surfaces cached artefacts, and logs the execution blueprint.
2. `sweeps` stage runs the hyper-parameter grid across learning rate, depth, estimators,
   regularisers, and text vectorisers.
3. `finalize` stage reloads the winning slate configuration per study, exports metrics,
   and optionally checkpoints models.
4. `reports` stage regenerates Markdown beneath `reports/xgb/`, including the opinion summaries.
   Opinion-regression sweeps run by default; use `--tasks` to target a subset (e.g. `--tasks next_video`).

Typical invocations:

```bash
# Emit a sweep plan summary without launching jobs
python -m xgb.pipeline --stage plan

# Dry-run to inspect cached vs pending tasks
python -m xgb.pipeline --dry-run

# Launch sweeps only with 8 parallel jobs
python -m xgb.pipeline --stage sweeps --jobs 8

# Finalize using cached sweeps (skips rerunning completed grids)
python -m xgb.pipeline --stage finalize --reuse-sweeps

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

### Defaults

The training launcher sets defaults that align with our standard sweeps:

- Tasks: `next_video,opinion` (set via `--tasks` or `XGB_PIPELINE_TASKS`).
- Text vectorisers: `tfidf,word2vec,sentence_transformer` (env `XGB_TEXT_VECTORIZER_GRID`).
- Sweep grids: `--learning-rate-grid 0.03,0.06`, `--max-depth-grid 3,4`,
  `--n-estimators-grid 250,350`, `--subsample-grid 0.8,0.9`,
  `--colsample-grid 0.7`, `--reg-lambda-grid 0.7`,
  `--reg-alpha-grid 0.1`.
- Tree method: uses GPU boosters by default; when GPUs are disabled or the
  installed `xgboost` lacks CUDA support, the launcher forces `--tree-method hist`.
- Sentence-Transformer device: `cuda` when GPUs are enabled; override with
  `SENTENCE_TRANSFORMER_DEVICE` or `XGB_SENTENCE_DEVICE`.
- GPU scheduling is enabled by default in the launcher (`XGB_USE_GPU=1`); set
  `XGB_USE_GPU=0` to force CPU-only execution.

## Feature Spaces & Vectorisers

- **TF-IDF** (default) – fast, memory-light baseline built via
  `core.vectorizers.TfidfVectorizerWrapper` (backed by `common.text.vectorizers`).
- **Word2Vec** – handled by `core.vectorizers.Word2VecVectorizer`; persisted models live
  under the directory supplied via `--word2vec_model_dir`.
- **Sentence Transformer** – implemented by `core.vectorizers.SentenceTransformerVectorizer`,
  which reuses the shared `common.text.embeddings` stack. Normalisation flags mirror the CLI
  options exposed through `common.cli.args`.

All vectorisers reuse `core.features.prepare_prompt_documents` to keep prompts aligned
with the other baselines.

## Opinion Workflow & Reports

`core.opinion` reuses the selected slate configuration to train regression models
per opinion study. Artefacts flow to `models/xgb/opinion/*`, while
`pipeline/reports/` produces Markdown under `reports/xgb/` (legacy imports against
`pipeline_reports/` continue to work via shims):

- `catalog.py` builds the README summary.
- `hyperparameter.py` publishes sweep tables and best-config recaps.
- `features.py` summarizes feature importance across slates.
- `opinion.py` renders opinion leadership tables and diagnostics.
- `plots.py` and `shared.py` house layout helpers used across the report modules.

Regenerate reports with `python -m xgb.pipeline --stage reports`.

### Task Overview

- **`next_video` (slate recovery):** Trains gradient-boosted recommenders to recreate the YouTube playlists observed in Liu et al. (_Short-term exposure to filter-bubble recommendation systems has limited polarization effects_, PNAS 2025). Matching held-out slates confirms the learned features capture the same cues as the production system before we study downstream impact.
- **`opinion` (opinion shift regression):** Uses participant-level pre/post indices to predict the post-study opinion score, reproducing the article’s measurement that shifts were small even under personalised feeds. This task exposes how model capacity and feature space choices influence the magnitude of predicted change.
- **`opinion_from_next` (exposure-informed shift):** Scores opinion change using the best `next_video` configuration so the identical reconstructed exposure path drives both predictions and counterfactuals. The reports tie these outputs back to the replication in `reports/research_article_political_sciences/README.md`, making the limited polarization findings easy to interpret alongside model diagnostics.

    **Leakage guard:** As of November 2025 the XGB opinion runner prunes training rows whose participant ID also exists in the validation slice before fitting. Earlier sweeps kept ~60 overlapping IDs from `data/cleaned_grail`, inflating MAE/R². Re-run sweeps/finals to refresh cached metrics with the stricter split.

### Opinion Shift Metrics

- **Participants:** `n_participants` (evaluation) and `train_participants` (training split) are written to `opinion_xgb_*_validation_metrics.json`; `eligible` mirrors the count of rows with finite before/after targets and model predictions and is the population used for every statistic.
- **Primary metrics:** `mae_after = mean(|ŷ_after - y_after|)`, `rmse_after = sqrt(mean((ŷ_after - y_after)^2))`, and `r2_after = 1 - Σ(y_after - ŷ_after)^2 / Σ(y_after - mean(y_after))^2`; `mae_change` and `rmse_change` evaluate the same formulas on opinion deltas where `ŷ_change = ŷ_after - y_before` and `y_change = y_after - y_before`.
- **Direction & calibration:** `direction_accuracy` measures how often the sign of `ŷ_change` matches `y_change` with a 1e-6 tolerance for “no change”; `calibration_slope`, `calibration_intercept`, and `calibration_bins` fit predicted change to observed change and back the expected calibration error `calibration_ece`; `kl_divergence_change` compares smoothed histograms of predicted vs. observed change.
- **Baseline comparison:** the `baseline` section scores a “no change” predictor (`pred_after = y_before`) with `mae_before`, `rmse_before`, `r2_before`, the change metrics ending in `_change_zero`, the calibration diagnostics ending in `_change_zero`, and the baseline `direction_accuracy` so deltas in the Markdown reports are interpretable.
- **Training curves:** `curve_metrics`, when present, records per-round MAE/RMSE for the train and validation splits (`mae_by_round`, `rmse_by_round`) and highlights the best validation round via `best_round` and `best_mae`; these values drive the opinion plots under `reports/xgb/opinion/`.
- **Per-participant exports:** predictions land in `opinion_xgb_*_validation_predictions.jsonl` with `prediction` (ŷ_after) and `prediction_change` (ŷ_change), making it easy to rebuild the same statistics offline if needed.

## Outputs & Testing

- Slate metrics, predictions, and curve snapshots land in
  `models/xgb/next_video/<vectorizer>/<study>/<issue>/`.
- Saved model bundles are serialised via `core.model.save_model_bundle`
  (still re-exported for legacy callers as `model.save_model_bundle`).
- Opinion metrics mirror the layout under `models/xgb/opinion/<study>/`.
- Opinion-from-next evaluations reuse the winning slate configuration; artefacts appear in
  `models/xgb/opinion/from_next/<vectorizer>/<study>/` and feed the optional
  `reports/xgb/opinion_from_next/` summaries.

Tests live in `tests/xgb/` and cover CLI glue, sweep planning, and vectoriser
behaviour. Extend them when adding new feature spaces or pipeline stages so CI
remains deterministic without access to the full dataset.

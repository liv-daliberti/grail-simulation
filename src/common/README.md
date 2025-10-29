# Common Pipeline Utilities

Shared building blocks used by the kNN, XGBoost, and GPT-4o pipelines. The
modules here provide consistent CLI ergonomics, prompt/document assembly, sweep
execution helpers, and reporting utilities.

## Module Overview

- **CLI helpers (`cli/`)**
  - `args.py` – reusable argparse builders for comma-separated flags,
    sentence-transformer options, and shared dataset arguments.
  - `options.py` – higher-level groups such as job-count, stage selection
    (`full`, `plan`, `sweeps`, `finalize`, `reports`), overwrite toggles, and logging configuration.

- **Prompt construction (`prompts/`)**
  - `docs.py` – canonical prompt builder that stitches together viewer
    profile, history, and slate context.
  - `fields.py` / `selection.py` – feature selection helpers and
    candidate metadata adapters shared by all baselines.
  - `sampling.py` – prompt down-sampling utilities used when
    sub-setting prompts for sweeps or reports.

- **Data adapters (`data/`)**
  - `hf_datasets.py` – optional Hugging Face dataset imports and environment helpers.

- **Pipeline execution (`pipeline/`)**
  - `stage.py` – task orchestration primitives (stage enums, dry-run
    summaries, execution logging).
  - `executor.py` / `utils.py` – fan-out helpers used by the kNN and XGBoost sweeps to
    parallelise CLI invocations.
  - `models.py`, `types.py`, `io.py` – dataclasses and
    I/O helpers for caching sweep results and metrics.
  - `formatters.py` – shared status-line helpers for sweep reporting.

- **Opinion & evaluation (`opinion/`, `evaluation/`)**
  - `models.py`, `metrics.py`, `sweep_types.py` – shared types,
    scoring functions, and aggregation helpers reused by both baselines.
  - `slate_eval.py`, `utils.py`, `matrix_summary.py` – metric calculations (accuracy, coverage,
    curve summaries) and dense/sparse previews consumed by `knn.core.evaluate` and `xgb.core.evaluate`.

- **Text & embeddings (`text/`)**
  - `vectorizers.py` – TF-IDF/Word2Vec/SentenceTransformer bundles with consistent
    persistence defaults.
  - `embeddings.py` / `utils.py` – sentence-transformer configuration,
    tokenisation helpers, and normalisation routines backing the vectorisers.
  - `title_index.py` – shared lookup helpers for resolving video titles.

- **Reporting (`reports/`, `visualization/`)**
  - `tables.py`, `fields.py`, `utils.py` – Markdown table builders and formatting helpers
    used across pipeline reports.
  - `visualization/matplotlib.py` – plotting defaults for optional report figures.

- **Auxiliary utilities**
  - `logging/utils.py` – structured logging configuration.
  - `import_utils.py` – safe, optional imports for heavy dependencies (sklearn, gensim, etc.).

- **Model-specific helpers (`ml/`)**
  - `xgb/callbacks.py`, `xgb/fit_utils.py` – shared fit-time callbacks and
    parameter harmonisation for the XGBoost baseline.

## Usage Notes

The modules are designed to be imported by downstream packages rather than run
directly. When extending the pipelines:

1. Prefer adding new CLI options in `common/cli/args.py` so they stay consistent
   across kNN and XGBoost.
2. Extend `prompts/docs.py` if additional prompt fields are required; the change
   will propagate to every baseline consumer.
3. Share new sweep or reporting dataclasses via `pipeline/types.py` to keep the
   stage executors generic.
4. Use `pipeline/stage.py`’s shared plan/sweeps/finalize/report helpers so new
   stages automatically benefit from dry-run summaries and task partitioning.

Tests covering these helpers live alongside the consumer packages (for example
`tests/knn/test_pipeline_modules.py`). Add targeted tests whenever you extend a
shared helper to avoid regressions across the baselines.

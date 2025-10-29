# Common Pipeline Utilities

Shared building blocks used by the kNN, XGBoost, and GPT-4o pipelines. The
modules here provide consistent CLI ergonomics, prompt document assembly, sweep
execution helpers, and reporting utilities.

## Module Overview

- **CLI helpers (`cli/`)**
  - `args.py` – reusable argparse builders for comma-separated flags,
    sentence-transformer options, and shared dataset arguments.
  - `options.py` – higher-level groups such as job-count, stage selection,
    overwrite toggles, and logging configuration.

- **Prompt construction (`prompts/`)**
  - `docs.py` – canonical prompt builder that stitches together viewer
    profile, history, and slate context.
  - `fields.py` / `selection.py` – feature selection helpers and
    candidate metadata adapters shared by all baselines.
  - `sampling.py` / `selection.py` – sampling utilities used when
    sub-setting prompts for sweeps or reports.

- **Vectorisers & embeddings (`text/`)**
  - `vectorizers.py` – lightweight wrappers around TF-IDF with consistent
    persistence defaults.
  - `embeddings.py` / `utils.py` – sentence-transformer configuration,
    tokenisation helpers, and normalisation routines that back the higher-level
    vectorisers.

- **Pipeline execution (`pipeline/`)**
  - `stage.py` – task orchestration primitives (stage enums, dry-run
    summaries, execution logging).
  - `executor.py` / `utils.py` – fan-out helpers used by kNN
    and XGBoost sweeps to parallelise CLI invocations.
  - `models.py`, `types.py`, `io.py` – dataclasses and
    I/O helpers for caching sweep results and metrics.

- **Opinion + metrics (`opinion/`, `evaluation/`)**
  - `models.py`, `metrics.py`, `sweep_types.py` – shared types,
    scoring functions, and aggregation helpers reused by both baselines.
  - `slate_eval.py`, `utils.py` – metric calculations (accuracy, coverage,
    curve summaries) consumed by `knn.evaluate` and `xgb.evaluate`.

- **Reporting (`reports/`, `visualization/`)**
  - `tables.py`, `utils.py` – Markdown table builders and formatting helpers
    used across pipeline reports.
  - `visualization/matplotlib.py` – plotting defaults for optional report
    figures.

- **Auxiliary utilities**
  - `logging/utils.py` – structured logging configuration.
  - `text/title_index.py` – shared lookup helpers for resolving video titles.

- **Model-specific helpers (`ml/`)**
  - `xgb/callbacks.py`, `xgb/fit_utils.py` – shared fit-time callbacks and
    parameter harmonisation for the XGBoost baseline.


- **Dataset adapters (`data/`)**
  - `hf_datasets.py` – optional Hugging Face dataset imports and environment
    helpers.

- **Shared evaluation tooling (`evaluation/`)**
  - `matrix_summary.py` – logging helpers for dense/sparse vector previews.

## Usage Notes

The modules are designed to be imported by downstream packages rather than run
directly. When extending the pipelines:

1. Prefer adding new CLI options in `common/cli/args.py` so they stay consistent
   across kNN and XGBoost.
2. Extend `prompts/docs.py` if additional prompt fields are required; the change
   will propagate to every baseline consumer.
3. Share new sweep or reporting dataclasses via `pipeline/types.py` to keep the
   stage executors generic.

Tests covering these helpers live alongside the consumer packages (for example
`tests/knn/test_pipeline_modules.py`). Add targeted tests whenever you extend a
shared helper to avoid regressions across the baselines.

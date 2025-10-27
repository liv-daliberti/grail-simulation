# Common Pipeline Utilities

Shared building blocks used by the kNN, XGBoost, and GPT-4o pipelines. The
modules here provide consistent CLI ergonomics, prompt document assembly, sweep
execution helpers, and reporting utilities.

## Module Overview

- **CLI helpers**
  - `cli_args.py` – reusable argparse builders for comma-separated flags,
    sentence-transformer options, and shared dataset arguments.
  - `cli_options.py` – higher-level groups such as job-count, stage selection,
    overwrite toggles, and logging configuration.

- **Prompt construction**
  - `prompt_docs.py` – canonical prompt builder that stitches together viewer
    profile, history, and slate context.
  - `prompt_fields.py` / `prompt_selection.py` – feature selection helpers and
    candidate metadata adapters shared by all baselines.
  - `prompt_sampling.py` / `prompt_selection.py` – sampling utilities used when
    sub-setting prompts for sweeps or reports.

- **Vectorisers & embeddings**
  - `vectorizers.py` – lightweight wrappers around TF-IDF, Word2Vec, and
    sentence-transformer encoders with consistent persistence APIs.
  - `embeddings.py` / `text.py` – tokenisation helpers and normalisation
    routines that back the higher-level vectorisers.

- **Pipeline execution**
  - `pipeline_stage.py` – task orchestration primitives (stage enums, dry-run
    summaries, execution logging).
  - `pipeline_executor.py` / `pipeline_utils.py` – fan-out helpers used by kNN
    and XGBoost sweeps to parallelise CLI invocations.
  - `pipeline_models.py`, `pipeline_types.py`, `pipeline_io.py` – dataclasses and
    I/O helpers for caching sweep results and metrics.

- **Opinion + metrics**
  - `opinion.py`, `opinion_metrics.py`, `opinion_sweep_types.py` – shared types,
    scoring functions, and aggregation helpers reused by both baselines.
  - `slate_eval.py`, `eval_utils.py` – metric calculations (accuracy, coverage,
    curve summaries) consumed by `knn.evaluate` and `xgb.evaluate`.

- **Reporting**
  - `report_tables.py`, `report_utils.py`, `matplotlib_utils.py` – Markdown table
    builders, formatting helpers, and plotting defaults used across pipeline
    reports.

- **Auxiliary utilities**
  - `logging_utils.py` – structured logging configuration.
  - `title_index.py` – shared lookup helpers for resolving video titles.

## Usage Notes

The modules are designed to be imported by downstream packages rather than run
directly. When extending the pipelines:

1. Prefer adding new CLI options in `common/cli_args.py` so they stay consistent
   across kNN and XGBoost.
2. Extend `prompt_docs.py` if additional prompt fields are required; the change
   will propagate to every baseline consumer.
3. Share new sweep or reporting dataclasses via `pipeline_types.py` to keep the
   stage executors generic.

Tests covering these helpers live alongside the consumer packages (for example
`tests/knn/test_pipeline_modules.py`). Add targeted tests whenever you extend a
shared helper to avoid regressions across the baselines.

# Pipeline Orchestration Helpers

`common.pipeline` captures the shared logic that underpins the baseline
pipelines (`knn`, `xgb`, `gpt4o`). It handles stage orchestration, job
fan-out, cache discovery, and shared dataclasses so each package only needs to
define task-specific pieces.

## Modules

- `stage.py` – enums and convenience functions for the canonical pipeline stages
  (`plan`, `sweeps`, `finalize`, `reports`) plus dry-run helpers.
- `executor.py` – worker pool and subprocess orchestration used when fanning out
  individual CLI invocations.
- `formatters.py` – status-line and log-format helpers for consistent CLI output.
- `io.py` – read/write helpers for cached metrics, sweep artifacts, and reports.
- `metrics.py`, `models.py`, `types.py` – dataclasses and type definitions that
  describe sweeps, evaluation results, and cached model bundles.
- `prompts.py` – shared prompt/document utilities consumed by the pipelines.
- `gpt4o_models.py` – glue for loading GPT-4o sweep artifacts; used to keep
  cross-baseline reports in sync.
- `utils.py` – miscellaneous helpers (path normalization, timestamp formatting,
  deterministic hashing).
- `__init__.py` – re-exports the most commonly used helpers.

Import from this package instead of re-implementing orchestration in each
baseline. When adding new pipeline stages or artifact types, extend the shared
dataclasses so all consumers stay aligned.

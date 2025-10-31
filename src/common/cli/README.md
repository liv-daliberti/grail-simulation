# Shared CLI Helpers

`common.cli` centralises argparse builders and option groups shared across the
baseline packages (`knn`, `xgb`, `gpt4o`) and RL trainers. Keeping the logic in
one place ensures consistent flag names, defaults, and help text.

## Modules

- `args.py` – low-level argument builders (dataset paths, comma-separated lists,
  sentence-transformer options, logging flags). These functions are composed by
  the downstream CLIs to assemble full parsers.
- `options.py` – higher-level option groups (job counts, stage selection, cache
  reuse toggles) tailored to the pipeline CLIs.
- `__init__.py` – re-exports the helper functions for convenient importing.

When adding a new CLI for a baseline, import from `common.cli` instead of
duplicating option definitions. Keep new flags backwards-compatible to avoid
breaking existing launch scripts.

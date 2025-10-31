# Dataset Helpers

`common.data` collects helper utilities for loading and validating datasets
used across the baselines and trainers. Keeping these routines in one place
avoids duplicating Hugging Face / local filesystem boilerplate.

## Modules

- `__init__.py` – exports convenience functions for external consumers.
- `hf_datasets.py` (if present) – optional helpers for interacting with
  Hugging Face datasets, including cache configuration and schema validation.

Additional dataset adapters should live here so downstream packages can reuse
them. Guard heavy or optional dependencies so import costs remain low.

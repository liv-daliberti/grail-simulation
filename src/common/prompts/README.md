# Prompt Assembly Helpers

`common.prompts` contains shared utilities for constructing prompt documents
from cleaned dataset rows. The helpers keep feature selection and formatting
consistent across baselines and RL trainers.

## Modules

- `fields.py` – defines the canonical prompt fields (viewer profile, watched
  history, slate metadata) and utilities to normalise missing values.
- `selection.py` – reusable feature-selection helpers that decide which prompt
  fields to include for a given experiment.
- `sampling.py` – routines for subsampling prompts when building evaluation
  cohorts or synthetic datasets.
- `__init__.py` – exports the high-level helpers for easy importing.

See `docs/api/common.prompts.rst` for generated API documentation. The prompt
builder package (`src/prompt_builder`) builds on these utilities to render
natural-language prompts.

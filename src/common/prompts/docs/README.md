# Prompt Document Builders

`common.prompts.docs` exposes programmatic builders for the structured prompt
representation used throughout the project. Whereas `prompt_builder` focuses on
rendering human-readable text, these modules emit dictionary-based payloads
that downstream models consume directly.

## Modules

- `builder.py` – entry points for assembling prompt documents from cleaned
  dataset rows; handles section ordering and optional fields.
- `extra_fields.py` – utilities for injecting experiment-specific metadata into
  prompt documents (e.g., study tags, fairness cohorts).
- `slate.py` – helpers for describing slate candidates (titles, channels,
  durations) in a structured form.
- `titles.py` – standardises title normalisation and fallback behaviour.
- `trajectory.py` – routines for capturing viewer watch histories as structured
  sequences.
- `__init__.py` – exports the builder helpers for import convenience.

Extend this package when new structured fields are required; keep the defaults
aligned with `prompt_builder` so text and structured prompts stay consistent.

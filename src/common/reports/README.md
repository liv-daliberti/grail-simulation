# Shared Report Builders

`common.reports` provides reusable helpers for assembling Markdown reports
across the project. Both baseline pipelines and RL trainers rely on these
utilities to generate consistent tables, fields, and formatting.

## Modules

- `fields.py` – reusable field descriptors and simple renderers for metrics,
  experiment settings, and provenance details.
- `tables.py` – opinionated Markdown table builders that handle column widths,
  footnotes, and alignment.
- `utils.py` – common formatting utilities (number formatting, path helpers,
  hyperlink generators) consumed by table/field builders.
- `__init__.py` – exports the main helper functions for easy importing.

When introducing new report sections, prefer extending these helpers so every
pipeline can share the improvements without duplicating rendering logic.

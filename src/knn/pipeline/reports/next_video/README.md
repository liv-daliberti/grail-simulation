# KNN Next-Video Reports

`knn.pipeline.reports.next_video` produces the slate-accuracy sections of the
pipeline reports. These modules convert evaluation artefacts into tables,
curves, and narrative summaries under `reports/knn/next_video/`.

## Modules

- `inputs.py` – loads cached evaluation outputs and prepares them for reporting.
- `comparison.py` – highlights performance differences across feature spaces,
  studies, and issue subsets.
- `curves.py` – formats MAE/accuracy-by-k curves for tables and embedded plots.
- `helpers.py` – shared formatting utilities and context builders.
- `loso.py` – leave-one-study-out analysis helpers.
- `report.py` – orchestrates the full next-video Markdown document.
- `sections.py` – section-specific renderers used by `report.py`.
- `csv_exports.py` – generates CSV artefacts for downstream analysis.
- `__init__.py` – exports the entry points.

Extend these modules when adding new diagnostics to the slate evaluation so the
reports remain comprehensive.

# KNN Pipeline Reports

`knn.pipeline.reports` turns cached sweep/final artifacts into Markdown reports
under `reports/knn/`. The package mirrors the structure used by the XGBoost
reports while tailoring content to kNN-specific outputs.

## Modules

- `catalog.py` – orchestrates report generation; loads cached metrics and
  delegates to section-specific builders.
- `features.py` – documents prompt feature spaces and configuration choices.
- `hyperparameter.py` – summarizes sweep grids, trial counts, and winning
  parameters.
- `opinion.py` / `opinion_sections.py` / `opinion_portfolio.py` /
  `opinion_csv.py` – opinion-shift reports, including CSV exports and portfolio
  summaries.
- `shared.py` – formatting helpers used across sections.
- `next_video/` – builders for the slate-accuracy report (see its README).

Extend the catalog when adding new report sections so the automation picks up
the additional content automatically.

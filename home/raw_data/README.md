# Raw Data for Portfolio Reproducibility

This folder is a self‑contained snapshot of the artifacts used to generate
`reports/main/README.md` (the Portfolio Comparison). Keeping this copy ensures
you can fully regenerate the portfolio table without rerunning training or
evaluation.

What’s included (mirrors original locations):

- `models/gpt-4o/next_video/**/metrics.json` — GPT‑4o next‑video study metrics.
- `models/grpo/**/next_video/**/metrics.json` — GRPO next‑video study metrics.
- `models/grail/**/next_video/**/metrics.json` — GRAIL next‑video study metrics.
- `models/{grpo,grail}/**/opinion/**/study*/metrics.json` — RLHF per‑study
  opinion metrics used for directional accuracy and MAE, plus baselines and N.
- Raw predictions and logs for reproducibility:
  - `models/{gpt-4o,grpo,grail}/**/next_video/**/predictions.jsonl`
  - `models/{gpt-4o,grpo,grail}/**/opinion/**/study*/predictions.jsonl`
  - `models/{grpo,grail}/**/opinion/**/study*/qa.log` (if present)
- `reports/knn/next_video/metrics.csv` and `reports/xgb/next_video/metrics.csv`
  — aggregated next‑video metrics for KNN and XGB.
- `reports/knn/opinion_from_next/opinion_metrics.csv` and
  `reports/xgb/opinion_from_next/opinion_metrics.csv` — opinion‑from‑next metrics
  for KNN and XGB.
- `reports/gpt4o/opinion/opinion_metrics.csv` — GPT‑4o opinion metrics and
  eligible counts used as fallback baselines.

How to regenerate the portfolio report from this snapshot:

1) Restore artifacts to their original locations (non‑destructive):

   - `rsync -a home/raw_data/models/ models/`
   - `rsync -a home/raw_data/reports/ reports/`

2) Rebuild the main report:

   - `python -m common.reports.portfolio`
     (writes `reports/main/README.md`)

Alternatively, you can run the higher‑level entrypoint that also refreshes
 per‑family pages when artifacts are present:

- `bash reports/build-reports.sh` (honors existing artifacts without retraining)

Provenance

- Files here were copied directly from the working tree (same relative paths)
  and listed in `home/raw_data/MANIFEST.txt` for auditing.

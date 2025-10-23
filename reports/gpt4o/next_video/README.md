# GPT-4o Next-Video Baseline

This report summarizes the promoted GPT-4o configuration from
`python -m gpt4o.pipeline --stage finalize`. When populated it contains:

- **Overall metrics** – accuracy, gold-in-slate coverage, formatting / parse
  success rate, and mean confidence for the validation split.
- **Fairness breakdowns** – tables grouped by issue and participant study so
  regressions in a single cohort surface quickly.
- **Telemetry** – pointers to the `metrics.json` and `predictions.jsonl` written
  under `models/gpt4o/<run>/`, which include the per-example traces and
  timestamps for auditing.

Regenerate this page after refreshing credentials or prompt templates:

```bash
PYTHONPATH=src python -m gpt4o.pipeline \
  --out-dir models/gpt4o \
  --reports-dir reports/gpt4o \
  --stage finalize \
  --stage reports
```

The command above reuses the last sweep winner; add `--stage sweeps` if you need
to rescore the candidate configurations first. When no evaluation has been run
the tables remain empty—populate them by running the finalize stage at least
once.

# GRPO Reports

The GRPO finetuning pipeline exports Markdown summaries under `reports/grpo/`.
Each run produces a pair of subdirectories:

- `next_video/` – next-video ranking metrics with issue and participant-study breakdowns.
- `opinion/` – opinion-regression diagnostics covering the canonical participant studies.

## Regenerating

```bash
python -m grpo.pipeline \
  --dataset data/cleaned_grail \
  --out-dir models/grpo \
  --label <run-label> \
  --stage reports
```

The helper script `reports/build-reports.sh` can re-run the reporting stage once
`models/grpo/<label>/` contains the cached evaluation artefacts. Set
`GRPO_REPORT_LABEL=<label>` to override the auto-detected run.

```{toctree}
:maxdepth: 1
:caption: GRPO Examples

grpo_sample_generative_responses
```

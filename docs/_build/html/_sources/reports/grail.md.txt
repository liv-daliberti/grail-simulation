# GRAIL Reports

The discriminator-augmented GRAIL runs mirror the GRPO layout and write their
Markdown summaries to `reports/grail/`.

- `next_video/` – next-video slate-selection accuracy, parsed/format rates, and fairness cuts.
- `opinion/` – opinion-regression metrics with per-study breakdowns vs. baseline.
- Shared configuration: the training dataset, prompt template, and base reward
  stack match the GRPO setup (for example `recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_gun.yaml`);
  the only differences are the checkpoint output location, hub id, and the
  optional GAIL shaping reward (toggled via `GAIL_USE`).

## Regenerating

```bash
python -m grail.pipeline \
  --dataset data/cleaned_grail \
  --out-dir models/grail \
  --label <run-label> \
  --stage reports
```

Set `GRAIL_REPORT_LABEL=<label>` before running `reports/build-reports.sh` to
refresh a specific evaluation run. The script automatically skips missing
artefacts when `REPORTS_ALLOW_INCOMPLETE=1`.

```{toctree}
:maxdepth: 1
:caption: GRAIL Baselines

grail_next_video
grail_opinion
grail_sample_generative_responses
```

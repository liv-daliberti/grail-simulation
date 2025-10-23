# GPT-4o Hyper-parameter Sweep

`python -m gpt4o.pipeline --stage sweeps` records every temperature /
max-token / top-p combination explored for the baseline. The generated Markdown
links to the leaderboard CSV and highlights whichever configuration advanced to
the finalize stage.

Each row in the table corresponds to a single call to `gpt4o.evaluate.run`,
showing:

- deployment metadata (model, temperature, max tokens, top-p);
- validation accuracy and coverage;
- formatting / parse success rate, which must stay near 100 %;
- mean latency and total token usage, useful for budgeting Azure quota.

Refresh the sweep after updating prompt templates or the dataset:

```bash
PYTHONPATH=src python -m gpt4o.pipeline \
  --out-dir models/gpt4o \
  --reports-dir reports/gpt4o \
  --stage sweeps \
  --stage reports
```

Skipping `--stage finalize` keeps the best configuration unchanged while still
publishing the new leaderboard.

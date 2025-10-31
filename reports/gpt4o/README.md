# GPT-4o Report Catalog

Generated artifacts for the GPT-4o slate-selection baseline:

- `next_video/` – summary metrics and fairness cuts for the selected configuration.
- `opinion/` – opinion-shift regression metrics across participant studies.
- `sample_generative_responses/README.md` – curated examples with the exact
  question prompts and the model's <think>/<answer> (and <opinion>) outputs,
  plus per-example notes.
- `hyperparameter_tuning/` – sweep results across temperature and max token settings.

Model predictions and metrics JSON files live under `models/gpt-4o/`.

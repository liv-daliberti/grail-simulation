# GRPO Report Catalog

Finetuned GRPO evaluation artifacts:

- `next_video/` – slate-ranking metrics for the configured checkpoint.
- `opinion/` – opinion regression metrics across participant studies.
- `sample_generative_responses/README.md` – curated examples showing the exact
  prompts given to the model and the model's <think>/<answer> (and <opinion>)
  outputs, with explanatory notes.

Regenerate via: python -m common.rlhf.aggregate_family_report --family grpo

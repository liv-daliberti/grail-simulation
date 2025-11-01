# GRAIL (Gun) Report Catalog

Finetuned GRAIL (Gun) evaluation artifacts:

- `next_video/` – slate-ranking metrics for the configured checkpoint.
- `opinion/` – opinion regression metrics across participant studies.
- `sample_generative_responses/README.md` – curated examples showing the exact
  prompts given to the model and the model's <think>/<answer> (and <opinion>)
  outputs, with explanatory notes.

Regenerate via `python -m grail.pipeline --stage full` after producing updated evaluation artifacts under `models/grail/`.

# GPT-4o Report Catalog

This directory mirrors the structure of the other baseline reports:

- `next_video/` – summary metrics and fairness cuts for the selected GPT-4o configuration.
- `hyperparameter_tuning/` – tables tracking the temperature/max-token sweep that produced the selection.

Run `python -m gpt4o.pipeline` (or `bash training/training-gpt4o.sh`) to regenerate these files after a new evaluation.


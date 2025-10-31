# RLHF Utilities

`common.rlhf` contains supporting helpers for reinforcement-learning-from-human
feedback workflows. The utilities are intentionally lightweight and shared by
the GRPO / GRAIL trainers when producing documentation artefacts.

## Modules

- `reports.py` – Markdown report builders that summarise RLHF training runs,
  including reward breakdowns and evaluation snapshots.
- `__init__.py` – exports the report helpers.

Extend this package when adding new RLHF-focused outputs so downstream trainers
and automation can stay aligned.

# GPT-4o Opinion Evaluation

`gpt4o.core.opinion` specialises the GPT-4o evaluation stack for opinion-shift
scoring. The helpers reuse baseline prompt construction but tailor the
conversation, settings, and aggregation for opinion metrics.

## Modules

- `settings.py` – dataclasses and defaults specific to opinion runs (e.g.,
  response temperature, scoring knobs).
- `helpers.py` – transforms cleaned rows into opinion-evaluation requests,
  handling prompt augmentation and pass-through metadata.
- `models.py` – structured containers for opinion predictions and metrics.
- `runner.py` – high-level execution helper that coordinates batched opinion
  evaluations and aggregates results.
- `__init__.py` – exports the public helpers.

Extend this package when adding new opinion diagnostics or changing evaluation
defaults so both the CLI and pipeline inherit the updates automatically.

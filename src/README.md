# Source Modules Overview

Everything under `src/` is importable project code: prompt rendering, prompting
baselines, and reinforcement-learning trainers all live here so they can share a
single packaging story.

## Packages

- `prompt_builder/` – canonical prompt/profile rendering helpers used by
  baselines and RL trainers (exposes `build_user_prompt`, `render_profile`, and
  related utilities).
- `knn/` – k-nearest-neighbour slate baseline with a modular CLI, Word2Vec/TF-IDF
  feature builders, and issue-specific dataset filtering helpers.
- `xgb/` – gradient-boosted tree baseline that mirrors the `knn` package
  structure but swaps the index for an XGBoost classifier.
- `gpt4o/` – Azure GPT-4o powered slate selector with a thin CLI wrapper and a
  shared conversation builder.
- `open_r1/` – GRPO/GRAIL trainers, SFT helpers, and recipe-driven configs used
  for reinforcement-learning experiments.
- `visualization/` – lightweight plotting utilities (e.g., recommendation tree
  rendering) imported by analysis notebooks and reporting scripts.

`prompt_builder.py` at the package root re-exports the modern package API for
callers that still import the legacy module.

## Development Guidelines

- Keep imports side-effect free; modules should be safe to import in tests and
  documentation builds.
- Treat external dependencies as optional unless they already appear in
  `requirements.txt` or `requirements-dev.txt`. Guard optional imports with
  helpful error messages.
- New public functions and classes require docstrings because Sphinx consumes
  these modules directly (`docs/api/**`).
- Add test coverage under `tests/` or the package-specific test suites when
  introducing new behaviour.
- Scripts that double as CLIs should expose a `main()` entry point guarded by
  `if __name__ == "__main__":` so they can be imported without immediately
  executing.

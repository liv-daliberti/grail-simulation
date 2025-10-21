# Source Modules Overview

Everything under `src/` is importable project code: prompt rendering, prompting
baselines, and reinforcement-learning trainers all live here so they can share a
single packaging story.

## Baseline Packages

- `knn/` – k-nearest-neighbor slate selector with modular CLIs, TF-IDF and
  optional Word2Vec features, and issue-aware dataset filtering. Typical run:
  `python -m knn.cli --dataset data/cleaned_grail --out_dir models/knn/run-001 --fit-index`.
- `xgb/` – XGBoost slate baseline that reuses the `knn` prompt/document pipeline
  but trains a multi-class booster. Train and evaluate via
  `python -m xgb.cli --fit_model --dataset data/cleaned_grail --out_dir models/xgb/run-001`.
- `gpt4o/` – Azure GPT-4o powered slate selection baseline with a structured
  conversation builder. After exporting API credentials, launch
  `python -m gpt4o.cli --out_dir reports/gpt4o --eval_max 100`.
- `open_r1/` – GRPO/GRAIL/SFT reinforcement-learning stack. Recipes under
  `recipes/**` feed the trainers; e.g.
  `python src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo.yaml`.

## Shared Utilities

- `prompt_builder/` – canonical prompt/profile rendering helpers used by
  baselines and RL trainers (exposes `build_user_prompt`, `render_profile`, and
  related utilities).
- `visualization/` – lightweight plotting utilities (e.g., recommendation tree
  rendering) imported by analysis notebooks and reporting scripts.
- `prompt_builder.py` – compatibility shim that re-exports the package API for
  legacy imports.

## Development Guidelines

- Keep imports side-effect free; modules should be safe to import in tests and
  documentation builds.
- Treat external dependencies as optional unless they already appear in
  `requirements.txt` or `requirements-dev.txt`. Guard optional imports with
  helpful error messages.
- New public functions and classes require docstrings because Sphinx consumes
  these modules directly (`docs/api/**`).
- Add test coverage under `tests/` or the package-specific test suites when
  introducing new behavior.
- Scripts that double as CLIs should expose a `main()` entry point guarded by
  `if __name__ == "__main__":` so they can be imported without immediately
  executing.

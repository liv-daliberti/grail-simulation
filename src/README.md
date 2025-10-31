# Source Modules Overview

Everything under `src/` is importable project code: prompt rendering, prompting
baselines, and reinforcement-learning trainers all live here so they can share a
single packaging story.

## Baseline Packages

- `knn/` – modular k-NN slate selector split into `cli/`, `core/`, and `pipeline/`
  packages. Run single experiments with
  `python -m knn.cli --dataset data/cleaned_grail --out_dir models/knn/run-001 --fit_index`
  or orchestrate the full workflow via
  `python -m knn.pipeline --stage {plan,sweeps,finalize,reports}`.
- `xgb/` – XGBoost slate baseline that mirrors the k-NN layout (`cli/`, `core/`,
  `pipeline/`) while reusing the shared prompt/document pipeline. Invoke
  `python -m xgb.cli --fit_model --dataset data/cleaned_grail --out_dir models/xgb/run-001`
  for a single run or `python -m xgb.pipeline --stage sweeps --jobs 8` for automation.
- `gpt4o/` – Azure GPT-4o powered slate-selection baseline with a structured
  conversation builder. After exporting API credentials, launch
  `python -m gpt4o.cli --out-dir reports/gpt4o --eval_max 100`
  or run the sweep/report harness via `python -m gpt4o.pipeline`.
- `common/open_r1/` – shared Open-R1 reinforcement-learning helpers (prompt
  preparation, reward wiring, dataset utilities) consumed by the GRPO and GRAIL
  trainers.
- `grpo/` – GRPO baseline training and evaluation entry points. Launch via
  `python src/grpo/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_gun.yaml`
  (swap `_gun` for `_wage` to target the wage task).
- `grail/` – GRPO + discriminator (GAIL-style) training entry points that extend
  the GRPO baseline with online reward shaping.

## Shared Utilities

- `common/` – shared CLI builders, prompt/document tooling, pipeline executors,
  reporting helpers, metrics, and text vectorisers consumed by every baseline.
- `prompt_builder/` – canonical prompt/profile rendering helpers used by
  baselines and RL trainers (exposes `build_user_prompt`, `render_profile`, and
  other helpers through a package namespace).
- `visualization/` – standalone recommendation-tree renderer CLI; plotting defaults
  shared with the baselines now live in `common/visualization/`.
- `prompt_builder.py` – compatibility shim that re-exports the package API for
  legacy imports.

## Development Guidelines

- Keep imports side-effect free; modules should be safe to import in tests and
  documentation builds.
- Treat external dependencies as optional unless they already appear in
  `requirements.txt` or `development/requirements-dev.txt`. Guard optional imports with
  helpful error messages.
- New public functions and classes require docstrings because Sphinx consumes
  these modules directly (`docs/api/**`).
- Add test coverage under `tests/` or the package-specific test suites when
  introducing new behavior.
- Scripts that double as CLIs should expose a `main()` entry point guarded by
  `if __name__ == "__main__":` so they can be imported without immediately
  executing.

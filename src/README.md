# Source Modules Overview

This directory contains the Python runtime code that augments the
cleaned GRAIL datasets with additional training utilities, evaluation
baselines, and prompt-rendering helpers. The key subpackages are:

- `prompt_builder/` – converts a cleaned interaction row into the
  natural-language profile and chat-style prompt used by downstream
  agents.
- `open_r1/` – reinforcement-learning scripts and configs for GRPO and
  related training recipes.
- `knn/` – non-generative baselines that share the same prompt-building
  logic for evaluation comparisons.

### Development Guidelines

- All modules under `src/` should be importable without side effects.
- Keep dependencies optional unless they are already required by
  `requirements.txt` or `requirements-dev.txt`; guard imports where
  possible.
- Any new public APIs should include docstrings ready for Sphinx
  auto-documentation.
- Add unit tests in `tests/` for each new module or behaviour.
- When introducing scripts, prefer a `main()` entry point guarded by
  `if __name__ == "__main__":` so the module can be imported safely.

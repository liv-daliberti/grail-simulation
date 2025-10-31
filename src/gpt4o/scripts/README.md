# GPT-4o Compatibility Script

The `gpt4o/scripts` folder stores a single backwards-compatible entry point:

- `gpt-4o-baseline.py` – thin wrapper that forwards to `gpt4o.cli.main`, letting
  legacy commands (`python src/gpt4o/scripts/gpt-4o-baseline.py`) continue to
  work after the module refactor.

Prefer running `python -m gpt4o.cli` or `python -m gpt4o.pipeline` directly for
new automation, and keep this script solely for legacy tooling.

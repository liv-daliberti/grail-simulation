# GPT-4o CLI Package

`gpt4o.cli` exposes the command-line interface for running GPT-4o-based slate
evaluations. Executing `python -m gpt4o.cli` launches the evaluation workflow,
mirroring the experience provided by the kNN and XGBoost CLIs.

## Modules

- `main.py` – argument parser and runtime driver; handles Azure credential
  configuration, dataset filtering flags, and evaluation options (temperature,
  top-p, max tokens).
- `__init__.py` – re-exports the CLI helpers for scripting.

Example invocation:

```bash
export PYTHONPATH=src
python -m gpt4o.cli \
  --out_dir models/gpt4o/example \
  --eval_max 100 \
  --top_p 0.95 \
  --temperature 0.7
```

Run `python -m gpt4o.cli --help` to inspect the full option set.

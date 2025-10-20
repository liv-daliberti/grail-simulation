# GPT-4o Slate Baseline

Modular implementation of the GPT-4o slate-selection baseline used across the
GRAIL simulation project. The original single-file script has been split into
focused modules so it mirrors the structure of the `knn/` and `xgb/` baselines.

## Quick start

1. **Configure Azure OpenAI** – either export credentials in your shell or edit
   `src/gpt4o/config.py`:

   ```bash
   export SANDBOX_API_KEY="..."
   export SANDBOX_ENDPOINT="https://api-ai-sandbox.princeton.edu/"
   export SANDBOX_API_VER="2025-03-01-preview"
   export DEPLOYMENT_NAME="gpt-4o"
   ```

2. **Install dependencies** (from the project root):

   ```bash
   pip install -r requirements-dev.txt
   ```

3. **Launch an evaluation**:

   ```bash
   python -m gpt4o.cli --out_dir reports/gpt4o --eval_max 100
   ```

   The CLI writes per-example predictions to `predictions.jsonl` and summary
   metrics to `metrics.json`. Pass `--deployment <name>` to override the default
   deployment configured in `config.py`.

`gpt4o.cli` downloads the cleaned dataset from Hugging Face (see
`config.DATASET_NAME`). Use `--cache_dir` to point at an existing HF cache or a
location with sufficient disk space. When disk pressure is detected the loader
automatically falls back to streaming mode.

> 💡 The legacy command `python src/gpt4o/gpt-4o-baseline.py` still works and now
> forwards to `gpt4o.cli:main`.

## Module layout

- `client.py` — thin wrapper around the Azure OpenAI client plus the `ds_call`
  helper used during evaluation.
- `config.py` — centralised configuration, dataset identifiers, and prompt
  template defaults.
- `conversation.py` — constructs messages from cleaned rows, including option
  formatting and answer-tag insertion for reliable parsing.
- `evaluate.py` — handles dataset loading, metrics aggregation, telemetry
  buckets, and the retry loop around API calls.
- `cli.py` — argument parser and runtime entry point (mirrors `knn`/`xgb` CLIs).
- `titles.py` / `utils.py` — shared helpers for title resolution, text
  canonicalisation, and response parsing.

Each module has a narrow responsibility, making it easy to swap components or
reuse the prompt builder in other workflows.

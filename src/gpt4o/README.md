# GPT-4o Slate Baseline

Modular implementation of the GPT-4o slate-selection baseline used across the GRAIL simulation project. The original single-file script has been split into focused modules so it mirrors the structure of the `knn/` and `xgb/` baselines.

## Quickstart

1. **Configure Azure OpenAI** – either set the sandbox variables in your shell or edit `src/gpt4o/config.py`:

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

3. **Run an evaluation**:

   ```bash
   python -m gpt4o.cli --out_dir reports/gpt4o --eval_max 100
   ```

   The CLI writes per-example predictions to `predictions.jsonl` and summary metrics to `metrics.json`.

> 💡 The legacy command `python src/gpt4o/gpt-4o-baseline.py` still works and now delegates to the modular CLI.

## Module Layout

- `client.py` — Thin wrapper around the Azure OpenAI client plus the `ds_call` helper.
- `config.py` — Centralised configuration, defaults, and prompt template.
- `conversation.py` — All prompt construction and viewer/profile formatting logic.
- `evaluate.py` — Streaming vs cached dataset loading and evaluation loop.
- `cli.py` — Argument parser and runtime entrypoint (mirrors `knn`/`xgb` CLIs).
- `titles.py` / `utils.py` — Shared helpers for title resolution and text canonicalisation.

Each module has a narrow responsibility, making it easier to swap components or reuse the prompt builder in other workflows.

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
   pip install -r development/requirements-dev.txt
   ```

3. **Launch an evaluation**:

   ```bash
   python -m gpt4o.cli --out_dir models/gpt-4o/debug --eval_max 100 --top_p 0.95
   ```

   The CLI writes per-example predictions to `predictions.jsonl` and summary
   metrics to `metrics.json`. Pass `--deployment <name>` to override the default
   deployment configured in `config.py`. Use `--dataset`, `--issues`, or
   `--studies` to override the source data or to slice the evaluation to a
   subset of issues/participant studies. Storing results in distinct subfolders
   (as shown above) keeps runs with different temperature / top-p / max-token
   settings from clobbering one another.

4. **Run the full pipeline** (hyper-parameter sweeps, final evaluation, and
   report regeneration):

```bash
   python -m gpt4o.pipeline --out-dir models/gpt-4o --reports-dir reports/gpt4o
   ```

   or invoke `bash training/training-gpt4o.sh` to reproduce the automation used
   by the other baselines. The pipeline mirrors the KNN/XGBoost workflows,
   sweeping temperatures, top-p values, and max-token caps, selecting the best
   configuration based on validation accuracy, and regenerating Markdown
   summaries (including fairness cuts by issue and participant study). After the
   next-video evaluation finishes, the same configuration is reused to score
   change-of-opinion at the participant level; the resulting artefacts live under
   `models/gpt-4o/opinion/<config>/` with summaries in `reports/gpt4o/opinion/`.

`gpt4o.cli` downloads the cleaned dataset from Hugging Face (see
`config.DATASET_NAME`). Use `--cache_dir` to point at an existing HF cache or a
location with sufficient disk space. When disk pressure is detected the loader
automatically falls back to streaming mode.

Curious how the cleaned rows are produced before uploading to Hugging Face?
`clean_data/sessions/README.md` walks through the session-ingestion pipeline
that backs every downstream baseline.

> 💡 The legacy command `python src/gpt4o/gpt-4o-baseline.py` still works and now
> forwards to `gpt4o.cli:main`.

## Module layout

- `client.py` — thin wrapper around the Azure OpenAI client plus the `ds_call`
  helper used during evaluation.
 - `config.py` — centralized configuration, dataset identifiers, and prompt
  template defaults.
- `conversation.py` — constructs messages from cleaned rows, including option
  formatting and answer-tag insertion for reliable parsing.
- `evaluate.py` — handles dataset loading, metrics aggregation, telemetry
  buckets, and the retry loop around API calls. It now records accuracy and
  formatting rates by issue and participant study for downstream fairness checks.
- `cli.py` — argument parser and runtime entry point (mirrors `knn`/`xgb` CLIs).
  Pass `--opinion_studies`, `--opinion_max_participants`, or
  `--opinion_direction_tolerance` to customise the opinion stage.
- `config.py` — aggregates shared settings, including the system prompts used for
  next-video ranking (`SYSTEM_PROMPT`) and opinion shift (`OPINION_SYSTEM_PROMPT`).
- `opinion.py` — GPT-4o opinion-shift evaluation runner shared by the pipeline.
- `pipeline.py` — orchestrates sweeps, selection, opinion evaluation, and report
  regeneration for the GPT-4o baseline.
- `titles.py` / `utils.py` — shared helpers for title resolution, text
  canonicalization, and response parsing.

Each module has a narrow responsibility, making it easy to swap components or
reuse the prompt builder in other workflows.

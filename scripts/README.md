# Scripts

Utility entrypoints for running validations, report generation, and data cleaning tasks from the command line.

- `scripts/run-tests.sh` — runs the Python test suite via `pytest` using `development/pytest.ini`. Accepts extra pytest flags. Sets up `PYTHONPATH` so in-repo packages resolve.
- `scripts/run-lint.sh` — executes `pylint` with the repo config but only enables error-level checks. Pass additional paths or pylint flags after the script name.
- `reports/build-reports.sh` — regenerates the KNN and XGBoost reports from existing sweep artifacts without rerunning training. Handles dataset discovery/assembly (local path, Hugging Face repo, or issue split assembly) before invoking `python -m knn.pipeline --stage reports` and `python -m xgb.pipeline --stage reports`. The legacy `scripts/run-build-reports.sh` wrapper now forwards directly to this entrypoint. Key env vars: `REPORTS_DATASET`, `REPORTS_ISSUE_DATASETS`, `KNN_REPORTS_*`, `XGB_REPORTS_*`. The pre-commit hook and the `Build Reports` GitHub Action both use this script so report markdown stays synced with the latest sweep artifacts while CI avoids training reruns.
- `scripts/run-clean-data-suite.sh` — end-to-end cleaning pipeline that reads a raw dataset, produces cleaned splits, and generates prompt statistics plus the political science replication report. Configurable through `GRAIL_*` env vars (e.g., source dataset, output directories, Hugging Face push settings, target issue repos).
- `scripts/update-reports.sh` — refreshes every report, including rebuilding the cleaned dataset, regenerating prompt samples, and (unless `GRAIL_SKIP_GPT4O=1`) executing the GPT-4o pipeline with `--overwrite`. Use this heavier workflow when you need to publish new data artifacts, not for the lightweight pre-commit checks.

All scripts are safe to execute from any location; they resolve the repository root internally before running.

## Git Hooks

The repository ships a version-controlled hook under `.githooks/pre-commit` that calls `reports/build-reports.sh` on every commit. Enable it once per clone:

```bash
git config core.hooksPath .githooks
```

Set `SKIP_REPORT_REFRESH=1` to bypass the hook for a single commit—for example, if you are touching documentation only or do not have the cleaned dataset available locally. Configure `REPORTS_DATASET` (or `REPORTS_ISSUE_DATASETS`) beforehand when the data lives outside the default `data/cleaned_grail` path.

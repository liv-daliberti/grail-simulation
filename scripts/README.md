# Scripts

Utility entrypoints for running validations, report generation, and data cleaning tasks from the command line.

- `scripts/run-tests.sh` — runs the Python test suite via `pytest` using `development/pytest.ini`. Accepts extra pytest flags. Sets up `PYTHONPATH` so in-repo packages resolve.
- `scripts/run-lint.sh` — executes `pylint` with the repo config but only enables error-level checks. Pass additional paths or pylint flags after the script name.
- `scripts/run-build-reports.sh` — orchestrates the KNN and XGBoost report builds. Handles dataset discovery/assembly (local path, Hugging Face repo, or issue split assembly), prepares output and cache directories, then runs `python -m knn.pipeline` and `python -m xgb.pipeline`. Key env vars: `REPORTS_DATASET`, `REPORTS_ISSUE_DATASETS`, `KNN_REPORTS_*`, `XGB_REPORTS_*`.
- `scripts/run-clean-data-suite.sh` — end-to-end cleaning pipeline that reads a raw dataset, produces cleaned splits, and generates prompt statistics plus the political science replication report. Configurable through `GRAIL_*` env vars (e.g., source dataset, output directories, Hugging Face push settings, target issue repos).

All scripts are safe to execute from any location; they resolve the repository root internally before running.

# KNN Compatibility Scripts

`knn.scripts` retains the historical entry point for the original kNN baseline
script. Modern code should invoke `python -m knn.cli` or `python -m knn.pipeline`,
but the script keeps older tooling working.

## Modules

- `baseline.py` – forwards to `knn.cli.main`, preserving the legacy argument
  surface for notebooks and SLURM jobs.
- `__init__.py` – empty marker file for packaging.

Prefer migrating callers to the modern CLIs when practical.

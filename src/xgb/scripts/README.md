# XGB Compatibility Scripts

`xgb.scripts` contains lightweight wrappers kept for backwards compatibility
with the original one-off baseline scripts. Modern code should import
`xgb.cli` or `xgb.pipeline` directly, but the scripts allow existing tooling to
keep working.

## Modules

- `baseline.py` – forwards legacy `python src/xgb/scripts/baseline.py` calls to
  `xgb.cli.main`. The script preserves the historical argument surface so old
  notebooks and SLURM jobs do not need to be rewritten.
- `__init__.py` – intentionally empty; marks the folder as a package.

Treat this package as deprecated glue—prefer updating callers to use the new
CLIs when convenient.

# KNN Pipeline Package

`knn.pipeline` automates the multi-stage workflow for the kNN slate baseline:
planning hyper-parameter jobs, executing sweeps, finalising best configurations,
and regenerating reports.

## Modules

- `cli.py` – exposes the pipeline CLI (`python -m knn.pipeline`) with stage
  selection, dry-run toggles, and job concurrency flags.
- `__main__.py` – enables module execution via `python -m knn.pipeline`.
- `context.py` – resolves datasets, model directories, and cache locations used
  during each stage.
- `sweeps.py` / `opinion_sweeps.py` – define the hyper-parameter grids and
  execution loops for next-video and opinion tasks.
- `evaluate.py` – reloads selected configurations, runs final evaluations, and
  exports artefacts.
- `io.py` – read/write helpers for sweep plans, cached metrics, and report
  inputs.
- `utils.py` – shared helpers (task partitioning, logging) leveraged by multiple
  stages.
- `reports/` – Markdown builders for `reports/knn` (see its README).

## Workflow

```bash
python -m knn.pipeline --stage plan
python -m knn.pipeline --stage sweeps --jobs 8
python -m knn.pipeline --stage finalize --reuse-sweeps
python -m knn.pipeline --stage reports --reuse-final
```

The pipeline tracks cached artefacts to avoid redundant compute. Extend
`context.py` and the sweep modules when introducing new feature spaces or tasks.

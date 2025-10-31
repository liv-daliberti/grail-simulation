# GPT-4o Pipeline Package

`gpt4o.pipeline` automates GPT-4o baseline experiments: planning sweeps,
executing evaluations, caching model outputs, and regenerating reports. Its
structure parallels the kNN/XGBoost pipelines.

## Modules

- `sweeps.py` – defines hyper-parameter grids (temperature, top-p, max tokens)
  and orchestrates execution across configurations.
- `models.py` – utilities for loading cached runs, promoting winners, and
  rehydrating conversation settings.
- `cache.py` – read/write helpers for evaluation artefacts (responses,
  metrics, prompts).
- `__main__.py` – allows execution via `python -m gpt4o.pipeline`.
- `__init__.py` – exports the public helpers.

Typical workflow:

```bash
python -m gpt4o.pipeline --out-dir models/gpt4o --reports-dir reports/gpt4o --stage plan
python -m gpt4o.pipeline --stage sweeps --jobs 4
python -m gpt4o.pipeline --stage finalize --reuse-sweeps
python -m gpt4o.pipeline --stage reports --reuse-final
```

The pipeline honours cached artefacts to minimise API usage; extend `cache.py`
when storing additional metadata.

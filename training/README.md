# Training Launchers

Shell helpers for running the baseline models and reinforcement-learning jobs
described in the project README.

## Scripts

- `training-grpo.sh` – launches the GRPO baseline with the ZeRO-3 accelerate
  profile and optional discriminator disabled.
- `training-grail.sh` – extends `training-grpo.sh` with the discriminator reward
  path used in the full GRAIL experiments.
- `training-knn.sh` – fits and evaluates the kNN slate baseline over the cleaned
  dataset.
- `training-xgb.sh` – runs the XGBoost slate baseline with optional model export
  for reuse across evaluation runs.

## Usage

All scripts assume the cleaned dataset lives at `data/cleaned_grail`. Override
paths or hyper-parameters via environment variables; any additional CLI flags
are forwarded to the underlying Python modules. Example:

```bash
DATASET=/path/to/dataset \
OUT_DIR=models/xgb/run-001 \
bash training/training-xgb.sh --issues minimum_wage
```

See `recipes/README.md` for the configuration files consumed by the GRPO and
GRAIL launchers.

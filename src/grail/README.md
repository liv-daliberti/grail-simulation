# GRAIL Trainer Package

`grail` implements the discriminator-augmented GRPO training loop used for the
GRAIL simulation experiments. The package extends the baseline GRPO trainer
(`src/grpo`) with an adversarial reward model (GAIL-style discriminator) and
extra wiring for joint policy / discriminator optimization.

## Module map

- `grail.py` – entry point for the full trainer; parses YAML recipes, wires
  together the actor, reward functions, discriminator, logging, and evaluation.
- `grail_dataset.py` – dataset preparation helpers specific to the
  discriminator setup (handles trajectory sampling, pass-through fields, and
  prompt assembly).
- `grail_gail.py` – discriminator model definitions plus utilities for training
  and checkpointing the adversarial reward component.
- `grail_mixer.py` – mixtures and sampling utilities for combining synthetic
  and human trajectories.
- `grail_rewards.py` – reward plumbing that combines GRPO base rewards with the
  discriminator score, including normalization and logging hooks.
- `grail_torch.py` – Torch-centric helpers (distributed setup, gradient clipping,
  device placement) used across the trainer.
- `grail_utils.py` – shared utilities (environment inspection, checkpoint I/O,
  logging helpers) that keep `grail.py` focused on orchestration.
- `pipeline.py` – command-line wrappers used by SLURM jobs and automation to
  kick off training runs with standard logging and artifact locations.
- `reports.py` – utilities that summarise training/evaluation metrics into the
  Markdown reports under `reports/grail/`.

All modules lean on the shared RL utilities in `common.open_r1`. When extending
the trainer, add generic helpers to `common.open_r1` so both `grpo` and `grail`
benefit.

## Usage

Launch training with a YAML recipe that specifies model, data, reward, and
logging settings:

```bash
python src/grail/grail.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail_gun.yaml
```

Swap `_gun` for `_wage` to target the wage task. Environment variables such as
`GAIL_WEIGHT`, `GAIL_ALPHA`, `GAIL_DISC_MODEL`, and `GRAIL_MAX_HISTORY` control
discriminator weighting, architecture, and prompt history depth. The helper
wrappers `training/training-grail-gun.sh` and `training/training-grail-wage.sh`
offer a turn-key launcher for SLURM.

## Extension checklist

1. Add new reward functions to `grail_rewards.py` and expose them through the
   YAML recipe schema.
2. Keep discriminator checkpoints lightweight—reuse the helpers in
   `grail_gail.py` for saving/loading so the pipeline scripts can surface model
   revisions automatically.
3. Update `reports.py` if you introduce new metrics or change logging paths; the
   automation expects consistent keys when regenerating Markdown.
4. Mirror improvements back into `grpo` where possible to avoid drift between
   the baseline and discriminator-enhanced trainers.

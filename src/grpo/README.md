# GRPO Trainer Package

`grpo` houses the vanilla GRPO reinforcement-learning trainer used as a
baseline in the GRAIL simulation project. The modules orchestrate dataset
prep, reward construction, policy updates, and reporting without the
discriminator path carried by `src/grail`.

## Module map

- `grpo.py` – main training entry point; parses YAML recipes, constructs the
  GRPO trainer, and coordinates evaluation/export.
- `config.py` – dataclass extensions and helpers for translating recipe YAML
  into structured configuration objects (augmenting TRL defaults).
- `dataset.py` – converts cleaned slates into GRPO-ready training/evaluation
  examples, handling filtering, batching, and optional mixtures.
- `model.py` – model loading and adapter utilities (LoRA, quantization, and
  Torch dtype/device shims) plus checkpoint management.
- `next_video.py` – task-specific evaluation helpers for slate accuracy.
- `opinion.py` – evaluation logic for opinion-shift scoring.
- `pipeline.py` – batch-friendly orchestration used by automation scripts; sets
  up logging directories, handles environment configuration, and launches
  accelerate when required.
- `reports.py` – Markdown report builders that summarise training/evaluation
  results under `reports/grpo/`.

Everything in this package shares prompt preparation, reward utilities, and CLI
conventions with `common.open_r1`. Add reusable building blocks there first so
both GRPO and GRAIL benefit.

## Usage

Start a training run with:

```bash
python src/grpo/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_gun.yaml
```

Adjust the recipe path for alternate models/tasks. Helpful toggles include
`--resume_from_checkpoint`, `--max_eval_samples`, and the environment variable
`LOGLEVEL=DEBUG` when debugging dataset filtering. The SLURM wrappers
`training/training-grpo-gun.sh` and `training/training-grpo-wage.sh` wire all
required environment variables for cluster runs.

## Extension checklist

1. Evolve reward logic in `common.open_r1.rewards` so the discriminator-enhanced
   trainer can reuse the same components.
2. Keep recipe schema changes backwards-compatible; report builders and
   automation expect stable keys.
3. Add tests under `tests/grpo/` (or extend existing fixtures) when introducing
   new evaluation modes or model backends.

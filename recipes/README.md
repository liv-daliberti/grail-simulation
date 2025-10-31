# Training Recipes

The `recipes/` directory collects configuration assets used to launch GRAIL
training jobs. Each YAML file is ingested by the TRL-powered entry points under
`src/grail/` and `src/grpo/`, with shared helpers living in
`src/common/open_r1/`, so updating hyperparameters, dataset mixtures, or logging
knobs rarely requires changing Python code.

## Layout

- `accelerate_configs/` – templates for `accelerate launch` (e.g. the Deepspeed
  ZeRO-3 config referenced by the SLURM wrappers in `training/`).
- `Qwen2.5-1.5B-Instruct/` – model-specific recipes. The `grpo/` folder holds
  GRPO and GRAIL (discriminator) configs used by `src/grpo/grpo.py` and
  `src/grail/grail.py`.

Add new model families by creating sibling directories (for example,
`Mistral-7B/`) and mirroring the task-oriented subfolders (`grpo/`, `sft/`,
`generate/`, etc.) as needed.

## Running a recipe

The trainers accept a `--config` flag pointing to any recipe file:

```bash
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
  src/grpo/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_gun.yaml
```

Swap `_gun` for `_wage` (and the analogous GRAIL recipe) to target the wage task.

The SLURM scripts in `training/` set the same environment variables (`CONFIG`,
`ACCEL_CONFIG`, `MAIN_SCRIPT`) before calling `accelerate launch`, so you can
override them to try alternate recipes without editing the shell scripts.

## Creating a new recipe

1. Copy the closest existing YAML file.
2. Update the `model`, `data`, and `trainer` sections, keeping output paths and
   logging destinations consistent with your experiment.
3. Commit the YAML under a descriptive subdirectory so follow-up runs can reuse
   it.

When adding a new accelerate config, drop the file in `accelerate_configs/` and
point `--config_file` or `ACCEL_CONFIG` at it. The job wrappers interpolate GPU
counts automatically, so most changes boil down to adjusting ZeRO stage, mixed
precision, or CPU offload knobs.

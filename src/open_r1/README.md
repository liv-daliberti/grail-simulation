# Open-R1 Reinforcement Learning Stack

`open_r1` holds the reinforcement-learning tooling used throughout the GRAIL
simulation project: prompt preparation, GRPO/GRAIL trainers, supervised
fine-tuning utilities, and generation pipelines for synthetic data.

## Module map

- `grpo.py` – vanilla GRPO trainer wired to TRL with recipe-driven configuration.
- `grail.py` – GRPO plus the online discriminator reward used in the GRAIL
  experiments (a GAIL-style extension).
- `sft.py` – supervised fine-tuning entry point sharing the same recipe shape.
- `generate.py` – helper for constructing distilabel pipelines that query vLLM or
  hosted OpenAI-compatible endpoints to produce training data.
- `configs.py` – dataclass extensions (`ScriptArguments`, `GRPOConfig`, `SFTConfig`)
  that add mixture support and logging knobs on top of TRL defaults.
- `rewards.py` – reward functions (accuracy, formatting checks, tool integration)
  plus helpers for interacting with external judging services.
- `utils/` – shared building blocks (code execution providers, IOI tools,
  dataset helpers) used by the reward functions and trainers.

Every script follows the same CLI shape exposed by TRL. Recipes under
`recipes/<model>/<task>/` supply configuration; the scripts simply parse those
YAMLs and translate them into trainer arguments.

## Prompt construction

All trainers rely on `prompt_builder.build_user_prompt`. Each row from the
cleaned dataset is converted into a multi-section message containing:

- `PROFILE:` – human-readable viewer summary synthesised from survey exports (see
  `clean_data/prompt/question_mapping.py` for the field mapping).
- `HISTORY:` – recent watches; governed by `GRAIL_MAX_HISTORY` (set to `0` for
  unbounded history).
- `CURRENT VIDEO:` – contextualises the slate being evaluated.
- `OPTIONS:` – candidate slate entries, optionally including YouTube IDs when
  `GRAIL_SHOW_IDS=1`.

The `_row_to_example` helpers in `grpo.py`/`grail.py` drop examples without a
resolvable gold click, normalise slate metadata, and surface pass-through fields
required by reward functions.

## Running GRPO

```bash
python src/open_r1/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo.yaml \
  --per_device_train_batch_size 2
```

Helpful CLI/environment toggles:

- `--resume_from_checkpoint <path>` resumes training.
- `--max_eval_samples` caps evaluation size when recipes enable validation.
- `LOGLEVEL=DEBUG` surfaces detailed dataset filtering and reward wiring info.
- `GRAIL_MAX_HISTORY` tunes the watch-history depth forwarded to the prompt
  builder.

The `training/training-grpo.sh` SLURM wrapper automates environment setup,
launches accelerate, and spins up a vLLM inference server when needed. Override
`CONFIG`, `MAIN_SCRIPT`, or accelerate-specific variables before submitting to
try alternate recipes.

## Running GRAIL (GRPO + discriminator)

```bash
python src/open_r1/grail.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail.yaml
```

Additional environment switches:

- `GAIL_USE=0` disables the discriminator path and falls back to pure GRPO.
- `GAIL_WEIGHT` rescales the discriminator reward before normalisation.
- `GAIL_ALPHA`, `GAIL_LR`, `GAIL_DISC_MODEL`, `GAIL_DEVICE` configure the
  discriminator backbone and optimiser.
- `GAIL_EVAL_MODE=1` can be exported to ensure the discriminator stays frozen
  during evaluation-only runs.

`training/training-grail.sh` mirrors the baseline launcher but wires in the
discriminator-specific variables.

## Supervised fine-tuning

`python src/open_r1/sft.py --config recipes/Qwen2.5-1.5B-Instruct/sft/config.yaml`
shares the same dataset mixture machinery as GRPO. Use `dataset_mixture` entries
in the recipe when combining multiple instruction sources; `configs.ScriptArguments`
validates column alignment across mixtures.

## Generation utilities

`generate.py` provides a `build_distilabel_pipeline` helper that packages a
distilabel `Pipeline` configured for OpenAI-compatible endpoints. Typical usage:

```bash
python src/open_r1/generate.py \
  --hf-dataset grail-sim/prompt-mixture \
  --model deepseek-r1 \
  --vllm-server-url http://localhost:8000/v1 \
  --prompt-template "{{ instruction }}"
```

Use this when you need to bootstrap preference data or evaluate alternative
prompt templates with large hosted models.

## Recipes & logging

All trainers load YAML recipes that share a common schema:

- **Model** – forwarded to TRL’s `get_model` helpers (`model_name_or_path`,
  `torch_dtype`, `attn_implementation`, PEFT config).
- **Data** – dataset identifiers, splits, system prompts, and optional mixtures.
- **Trainer knobs** – batch sizes, gradient checkpointing, eval cadence, save
  cadence, logging frequency.
- **Rewards** – names mapped to callables in `rewards.py` with optional weights;
  weights are normalised automatically.
- **Reproducibility / Hub** – seeds, `push_to_hub`, hub repo IDs, and revisions.

Metrics stream to stdout and Weights & Biases when the relevant environment
variables or recipe entries (`wandb_project`, `wandb_entity`) are set. Local logs
land in `logs/train_*`; accelerator outputs and checkpoints follow the recipe’s
`output_dir`.

## Workflow checklist

1. Prepare datasets via `clean_data/clean_data.py` so required columns (prompt,
   slate metadata, gold click) are populated.
2. Pick or duplicate a recipe under `recipes/**` and adjust model, data, and
   reward settings.
3. Launch `grpo.py`, `grail.py`, or `sft.py` locally or through the SLURM
   wrappers in `training/`.
4. Monitor trainer logs, W&B dashboards, and generated checkpoints under
   `output_dir`.
5. Iterate on reward configuration or prompt design as needed—no code changes
   are required for most experiments thanks to the recipe-driven approach.

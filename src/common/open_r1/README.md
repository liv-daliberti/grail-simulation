# Open-R1 Reinforcement Learning Stack

`common.open_r1` holds the reinforcement-learning utilities shared across the
GRAIL simulation project: prompt preparation, configuration helpers, reward
functions, and dataset tooling used by both the GRPO and GRAIL training loops.
The concrete training entry points now live in `src/grpo/` and `src/grail/`,
while this package provides the shared building blocks they consume.

## Module map

- `configs.py` – dataclass extensions (`ScriptArguments`, `GRPOConfig`) that add
  mixture support and logging knobs on top of TRL defaults.
- `dataset_utils.py` – routines for filtering and validating GRPO-compatible
  slates.
- `example_utils.py` – helpers that convert cleaned rows into training examples.
- `generate.py` – helper for constructing distilabel pipelines that query vLLM or
  hosted OpenAI-compatible endpoints to produce training data.
- `pure_accuracy_utils.py` – utilities supporting the accuracy-based rewards.
- `rewards.py` – reward functions (accuracy, formatting checks, lightweight code
  evaluation) plus helpers for logging reward diagnostics.
- `shared.py` – prompt assembly, passthrough field wiring, and GRPO pipeline
  orchestration helpers consumed by both trainers.
- `torch_stub_utils.py` – lightweight Torch shims for environments without GPU
  support.
- `utils/` – callbacks, replay buffers, dataset loaders, and push-to-hub helpers
  used by the reward functions and trainers.
- Training entry points – `src/grpo/grpo.py` (baseline) and `src/grail/grail.py`
  (discriminator-augmented) import the modules above to assemble full pipelines.

Every script follows the same CLI shape exposed by TRL. Recipes under
`recipes/<model>/<task>/` supply configuration; the scripts simply parse those
YAMLs and translate them into trainer arguments.

## Prompt construction

All trainers rely on `prompt_builder.build_user_prompt`. Each row from the
cleaned dataset is converted into a multi-section message containing:

- `VIEWER` – single-line viewer summary synthesized from survey exports (see
  `clean_data/prompt/question_mapping.py` for the field mapping).
- `Initial Viewpoint` – optional issue-specific stance synthesized from the survey.
- `CURRENTLY WATCHING` – contextualizes the slate being evaluated.
- `RECENTLY WATCHED (NEWEST LAST)` – recent watches; governed by `GRAIL_MAX_HISTORY`
  (set to `0` for unbounded history).
- `SURVEY HIGHLIGHTS` – optional recap of salient survey responses.
- `OPTIONS` – candidate slate entries, optionally including YouTube IDs when
  `GRAIL_SHOW_IDS=1`.

The `_row_to_example` helpers in `grpo.py`/`grail.py` drop examples without a
resolvable gold click, normalize slate metadata, and surface pass-through fields
required by reward functions.

## Running GRPO

```bash
python src/grpo/grpo.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo_gun.yaml \
  --per_device_train_batch_size 2
```

Swap `_gun` for `_wage` in the recipe path to target the wage task.

Helpful CLI/environment toggles:

- `--resume_from_checkpoint <path>` resumes training.
- `--max_eval_samples` caps evaluation size when recipes enable validation.
- `LOGLEVEL=DEBUG` surfaces detailed dataset filtering and reward wiring info.
- Set `push_to_hub_revision: true` (and `hub_model_id`) in the YAML recipe to
  mirror every saved checkpoint to the configured Hugging Face Hub repo.
- `GRAIL_MAX_HISTORY` tunes the watch-history depth forwarded to the prompt
  builder.

The `training/training-grpo.sh` SLURM wrapper automates environment setup,
launches accelerate, and spins up a vLLM inference server when needed. Override
`CONFIG`, `MAIN_SCRIPT`, or accelerate-specific variables before submitting to
try alternate recipes.

## Running GRAIL (GRPO + discriminator)

```bash
python src/grail/grail.py \
  --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail_gun.yaml
```

Use the `_wage` variant of the recipe for wage-specific runs.

Additional environment switches:

- `GAIL_USE=0` disables the discriminator path and falls back to pure GRPO.
- `GAIL_WEIGHT` rescales the discriminator reward before normalization.
- `GAIL_ALPHA`, `GAIL_LR`, `GAIL_DISC_MODEL`, `GAIL_DEVICE` configure the
  discriminator backbone and optimizer.
- `GAIL_EVAL_MODE=1` can be exported to ensure the discriminator stays frozen
  during evaluation-only runs.

`training/training-grail.sh` mirrors the baseline launcher but wires in the
discriminator-specific variables.

## Generation utilities

`generate.py` provides a `build_distilabel_pipeline` helper that packages a
distilabel `Pipeline` configured for OpenAI-compatible endpoints. Typical usage:

```bash
python -m common.open_r1.generate \
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
  weights are normalized automatically.
- **Reproducibility / Hub** – seeds, `push_to_hub`, hub repo IDs, and revisions.

Metrics stream to stdout and Weights & Biases when the relevant environment
variables or recipe entries (`wandb_project`, `wandb_entity`) are set. Local logs
land under `${LOG_DIR:-logs/grpo}` (for example `logs/grpo/train_*` when using
`training-grpo.sh`); accelerator outputs and checkpoints follow the recipe’s
`output_dir`.

## Workflow checklist

1. Prepare datasets via `clean_data/clean_data.py` so required columns (prompt,
   slate metadata, gold click) are populated.
2. Pick or duplicate a recipe under `recipes/**` and adjust model, data, and
   reward settings.
3. Launch `src/grpo/grpo.py` or `src/grail/grail.py` locally or through the
   SLURM wrappers in `training/`.
4. Monitor trainer logs, W&B dashboards, and generated checkpoints under
   `output_dir`.
5. Iterate on reward configuration or prompt design as needed—no code changes
   are required for most experiments thanks to the recipe-driven approach.

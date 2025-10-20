# Open-R1 Reinforcement Learning Trainers

This submodule contains the reinforcement-learning tooling used in the GRAIL simulation project. It ships:

- **Prompt authoring** utilities (the `prompt_builder` package) that turn raw viewer logs into model-ready chat prompts.
- **Baseline GRPO training** (`grpo.py`) which mirrors the vanilla TRL pipeline.
- **GRAIL + discriminator training** (`grail.py`) which augments GRPO with an online GAIL-style reward.

The sections below describe how prompts are constructed, how each trainer consumes them, the configuration knobs exposed through the YAML recipes, and how to launch end-to-end runs.

## Prompt construction pipeline

`build_user_prompt` is the entry point used by both trainers. Each dataset row is converted into a multi-section chat message:

- `PROFILE:` is a single paragraph synthesised from viewer demographics, political attitudes, minimum-wage beliefs, and media habits. The helper collapses structured fields into natural-language sentences while guaranteeing at least one sentence is emitted for every row (falls back to “Profile information is unavailable.”).
- `HISTORY (most recent first):` lists prior watches with watch-time metadata when history exists. The `GRAIL_MAX_HISTORY` environment variable controls how many past items are shown; set `GRAIL_MAX_HISTORY=0` to include the full history.
- `CURRENT VIDEO:` includes the currently playing video to provide recency when the slate is drawn.
- `OPTIONS:` enumerates the candidate slate, pairing the title (or fallback id) with channel and duration. When running with `GRAIL_SHOW_IDS=1`, item ids are also surfaced to help debugging.

The prompt builder never relies on session context anymore: only the sections above are emitted, ensuring a consistent prompt surface for downstream models.

The helper is used inside `_row_to_example` in both `grpo.py` and `grail.py`. That function also:

1. Cleans and de-duplicates slate metadata via `_load_slate_items`.
2. Maps the gold click to a 1-indexed label (`gold_index`) or drops the row when no match exists.
3. Preserves pass-through features (`viewer_profile`, `slate_items`, history JSON) used by reward functions and logging.

## GRPO baseline training (`src/open_r1/grpo.py`)

The GRPO script implements a vanilla TRL loop:

1. **Dataset intake** – `get_dataset` loads the Hugging Face dataset described in the recipe. Rows without a usable slate or gold click are filtered eagerly.
2. **Prompt mapping** – each row is transformed via `_row_to_example`, and optional `__drop__` flags from data preprocessing are honoured.
3. **Column pruning** – only the fields consumed by rewards and the trainer are retained, keeping the dataset lean for accelerator workers.
4. **Reward wiring** – `get_reward_funcs` translates the recipe’s `reward_funcs` list into callables. `reward_weights` are normalised so any YAML magnitudes sum to 1.0.
5. **Evaluation gating** – when `do_eval: true`, the script enforces `evaluation_strategy: steps` and validates `eval_steps > 0`. You can optionally cap validation size with `max_eval_samples`.
6. **Training** – `GRPOTrainer` from TRL is instantiated with the model, tokenizer, datasets, and PEFT configuration. If a checkpoint exists in `output_dir` (or is specified via `resume_from_checkpoint`), training resumes automatically.

The script reads the same arguments as the YAML recipe through `TrlParser`, so adding parameters to `recipes/.../config_grpo.yaml` is all that’s needed for most experiments.

### Launching a run

The simplest way to start the baseline is via the provided SLURM wrapper:

```bash
sbatch training/training-grpo.sh
```

This script provisions a fresh environment, spins up vLLM serving for rollouts, and launches Accelerate with the YAML located at `recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo.yaml`. Override `CONFIG`, `MAIN_SCRIPT`, or other environment variables when submitting to try new recipes.

For local experiments you can invoke the trainer directly (no distributed setup):

```bash
python src/open_r1/grpo.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grpo.yaml
```

Any CLI flag accepted by the YAML (e.g., `--per_device_train_batch_size 2`) can be appended to override specific fields.

## GRAIL + discriminator training (`src/open_r1/grail.py`)

The GRAIL script extends the GRPO baseline with an optional online discriminator that supplies an auxiliary reward:

1. **Shared preprocessing** – dataset filtering, prompt construction, and reward lookup mirror the GRPO script.
2. **Discriminator bootstrap** – when `GAIL_USE` is truthy (default `1`), an `OnlineDiscriminator` is created. Helper `_pick_disc_device` aligns the discriminator device with the local rank to support distributed training.
3. **Reward augmentation** – `make_gail_reward_fn` wraps discriminator scores into the GRPO reward interface. When enabled, the wrapper adds an extra reward function (`gail_reward`) and optionally extends `reward_weights` with `GAIL_WEIGHT`.
4. **Training loop** – `GRPOTrainer` runs as usual, but the GAIL reward is applied to completions. During evaluation, `GAIL_EVAL_MODE` is toggled to prevent discriminator updates.

### Key environment toggles

The SLURM launcher (`training/training-grail.sh`) exposes several environment variables you can export before running:

- `GAIL_USE` (default `1`) – disable to fall back to pure GRPO while still using the GRAIL script.
- `GAIL_WEIGHT` – scales the discriminator reward relative to others; weights are re-normalised automatically.
- `GAIL_ALPHA`, `GAIL_LR`, `GAIL_DISC_MODEL`, `GAIL_DEVICE` – control the discriminator temperature, learning rate, backbone, and placement.
- `GRAIL_MAX_HISTORY` – forwarded to the prompt builder to adjust watch-history depth.

The rest of the training flow (evaluation, checkpointing, push-to-hub) follows the GRPO baseline.

### Launching a run

```bash
sbatch training/training-grail.sh
```

As with the baseline, override `CONFIG` to point at `recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail.yaml` variations or supply alternate accelerate configs.

To run locally without SLURM:

```bash
python src/open_r1/grail.py --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_grail.yaml
```

Remember to export any GAIL-related environment variables beforehand if you want to customise the discriminator behaviour.

## Recipe anatomy

Both trainers consume the same YAML schema (see `recipes/Qwen2.5-1.5B-Instruct/grpo/`):

- **Model section** (`model_name_or_path`, `torch_dtype`, `attn_implementation`) – forwarded to `get_model`.
- **Data section** (`dataset_name`, `dataset_*_split`, `dataset_solution_column`, `system_prompt`) – decides which HF dataset and prompt template to load.
- **Trainer knobs** (`per_device_*_batch_size`, `gradient_checkpointing`, `num_train_epochs`, `evaluation_strategy`, `eval_steps`, `save_steps`, `log_*`) – copied into `GRPOConfig`, then passed straight to TRL’s `GRPOTrainer`.
- **Rewards** (`reward_funcs`, `reward_weights`) – interpreted by `get_reward_funcs` and normalised in-script.
- **KL / PPO controls** (`kl_target`, `clip_range`, `vf_coef`, etc.) – directly exposed by `GRPOConfig`, so they can be tuned per recipe with no code changes.
- **Hub + reproducibility** (`seed`, `push_to_hub`, `hub_model_id`) – respected by both scripts; push-to-hub is invoked automatically at the end of training when enabled.

## Putting it all together

1. **Prepare data** with `clean_data/clean_data.py` to ensure the splits contain `slate_items_json`, `viewer_profile_sentence`, and the target solution column.
2. **Adjust the recipe** under `recipes/.../grpo/` to set batch sizes, evaluation cadence, reward mix, and system prompt text.
3. **Choose a trainer**: use `grpo.py` for the baseline or `grail.py` when you want discriminator shaping.
4. **Launch via SLURM** (`training/training-grpo.sh` or `training/training-grail.sh`) or run the Python entrypoint directly for small-scale tests.
5. **Monitor logs** in `logs/train_*` (trainer metrics) and `.wandb/` if Weights & Biases logging is enabled. The scripts also log the resolved reward functions and weights at startup.

With this workflow you can iterate on prompt design, reward shaping, and GAIL hyper-parameters without modifying the core trainers—most behaviour is driven by the YAML recipes and environment variables described above. 

For a concrete run-through featuring both trainers, refer to the accompanying [Weights & Biases report](https://wandb.ai/ogd3-princeton-university/grail/reports/GRAIL--VmlldzoxNDc2NDk2Mw?accessToken=t4hs02nleorgeuklpjtzganan4pizwxry6ld2c44t09nfxrmv1woks2xiy6fh0zy)

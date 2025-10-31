# Open-R1 Utility Suite

`common.open_r1.utils` gathers supporting utilities for the shared RL stack.
These helpers cover callbacks, dataset management, Hugging Face Hub uploads,
and replay buffers. Both `grpo` and `grail` trainers import from this package.

## Modules

- `callbacks.py` – trainer callbacks for logging rewards, monitoring gradients,
  and integrating with external trackers (e.g., Weights & Biases).
- `data.py` – dataset loading pipelines, including mixture handling and schema
  validation.
- `evaluation.py` – helpers that convert trainer outputs into evaluation
  summaries and metrics dictionaries.
- `hub.py` – wrappers around Hugging Face Hub uploads/downloads used to push
  checkpoints or fetch reference models.
- `model_utils.py` – Torch-centric utilities (parameter freezing, dtype/device
  helpers) that keep training scripts concise.
- `replay_buffer.py` – lightweight replay buffer implementation for preference
  data and batched rollouts.
- `__init__.py` – exports the public helpers.

Extend these modules when introducing new trainer integrations so both GRPO and
GRAIL pipelines benefit without duplicating boilerplate.

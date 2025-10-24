#!/usr/bin/env python
"""Trainer callbacks that integrate Open-R1 runs with auxiliary services.

The callbacks defined here extend Hugging Face ``Trainer`` runs with behaviour
used by the Open-R1 experiments: automatically pushing checkpoints to the Hub,
collecting high-performing prompts in a replay buffer, and mirroring batches
into that buffer in real time.  The functions are written to be picked up by
Sphinx autodoc so they show up in the generated API reference.
"""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import Future
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Sequence

from open_r1.utils.replay_buffer import ReplayBuffer

try:  # pragma: no cover - optional dependency
    from transformers import (  # pylint: disable=import-error
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
except ImportError as exc:  # pragma: no cover
    class TrainerCallback:  # type: ignore[too-few-public-methods]  # pylint: disable=too-few-public-methods
        """Fallback that surfaces a helpful error when Transformers is missing."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError(
                "transformers is required to use Open-R1 Trainer callbacks. "
                "Install it with `pip install transformers`."
            ) from exc

    class TrainerControl:  # type: ignore[too-few-public-methods]  # pylint: disable=too-few-public-methods
        """Transformers control placeholder used when the package is unavailable."""

    class TrainerState:  # type: ignore[too-few-public-methods]  # pylint: disable=too-few-public-methods
        """Transformers state placeholder used when the package is unavailable."""

    class TrainingArguments:  # type: ignore[too-few-public-methods]  # pylint: disable=too-few-public-methods
        """Transformers arguments placeholder used when the package is unavailable."""

# ---------------------------------------------------------------------------
#  SLURM helper --------------------------------------------------------------
# ---------------------------------------------------------------------------

def _slurm_available() -> bool:
    """Return ``True`` when the `sinfo` binary is available (SLURM)."""

    try:
        subprocess.run(
            ["sinfo"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except FileNotFoundError:
        return False

# ---------------------------------------------------------------------------
#  Push-to-hub callback ------------------------------------------------------
# ---------------------------------------------------------------------------

class PushToHubRevisionCallback(TrainerCallback):
    """Callback that pushes checkpoints to the Hub using revision tags."""

    def __init__(self, model_cfg):
        """Store the model configuration used when scheduling pushes."""
        super().__init__()
        self.model_cfg = model_cfg
        self.log = logging.getLogger("PushToHub")

    def on_save(  # type: ignore[override]  # signature must match TrainerCallback
        self,
        args: TrainingArguments,
        state: TrainerState,
        _control: TrainerControl,
        **_kwargs,
    ) -> None:
        """Push the current checkpoint to the Hub and optionally schedule benchmarks."""
        if not state.is_world_process_zero:
            return

        step_tag = f"step-{state.global_step:09d}"
        dummy = SimpleNamespace(
            benchmarks=None,
            hub_model_id=args.hub_model_id,
            hub_model_revision=f"{args.hub_model_revision}-{step_tag}",
            output_dir=f"{args.output_dir}/checkpoint-{state.global_step}",
            system_prompt=args.system_prompt,
        )

        # lazy import – avoids circular deps if huggingface_hub absent
        from .hub import push_to_hub_revision  # pylint: disable=import-outside-toplevel
        fut: Future = push_to_hub_revision(dummy, extra_ignore_patterns=["*.pt"])

        # (optional) spawn benchmark job when the upload finishes
        if _slurm_available():
            def _after(_):
                """Submit benchmark jobs once the upload future resolves.

                :param _: Completed future returned by the upload helper.
                """
                from .evaluation import run_benchmark_jobs  # pylint: disable=import-outside-toplevel
                self.log.info("Upload done – submitting benchmark job.")
                dummy.benchmarks = args.benchmarks
                run_benchmark_jobs(dummy, self.model_cfg)
            fut.add_done_callback(_after)  # pylint: disable=no-member

# ---------------------------------------------------------------------------
#  Success-caching callback (text-log scraper) -------------------------------
# ---------------------------------------------------------------------------

class SuccessCachingCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """
    Scrape ``trainer._textual_logs`` after each log step and push any prompt whose
    accuracy meets or exceeds ``acc_threshold`` into the attached ``ReplayBuffer``.

    Note:
        Transformers never passes the trainer instance via ``**kwargs``; call
        :meth:`set_trainer` during setup to register it.
    """

    def __init__(self, replay_buffer: ReplayBuffer, acc_threshold: float = 0.999):
        """Initialise the callback with a replay buffer and accuracy threshold."""
        super().__init__()
        self.buf = replay_buffer
        self.thr = acc_threshold
        self._trainer = None                         # will be set later
        self.log = logging.getLogger("SuccessCache")

    # ---------- lifecycle hooks ------------------------------------------
    def set_trainer(self, trainer):                  # called once at start
        """Register the owning trainer instance for later log inspection."""
        self._trainer = trainer

    # ---------- main hook -------------------------------------------------
    def on_log(
        self,
        _args: TrainingArguments,
        _state: TrainerState,
        _control: TrainerControl,
        _logs: Optional[Dict[str, float]] = None,
        **_kwargs,
    ) -> None:
        """Scrape textual logs and add high-accuracy prompts to the buffer."""
        if self._trainer is None:
            return

        txt_logs = getattr(self._trainer, "_textual_logs", None)
        if not isinstance(txt_logs, Mapping):
            return

        prompts = txt_logs.get("prompt")
        rewards = txt_logs.get("rewards")
        if not (
            isinstance(rewards, Mapping)
            and isinstance(prompts, Sequence)
            and not isinstance(prompts, (str, bytes))
            and prompts
        ):
            return

        # pick the accuracy reward head (name may differ in your config)
        acc_key = next((k for k in rewards if "accuracy" in k), None)
        if acc_key is None:
            return

        accuracy_series = rewards.get(acc_key)
        if not (
            isinstance(accuracy_series, Sequence)
            and not isinstance(accuracy_series, (str, bytes))
        ):
            return

        for prompt, acc in zip(prompts, accuracy_series):
            if acc >= self.thr:
                self.buf.add(prompt)

# ---------------------------------------------------------------------------
#  Replay-buffer callback (fast path – uses training_step outputs) ----------
# ---------------------------------------------------------------------------

class ReplayBufferCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """Callback that logs batches directly into the replay buffer."""

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        tokenizer,
        accuracy_key: str = "crossword_accuracy_reward",
        threshold: float = 1.0,
    ):
        """Configure the callback with replay buffer, tokenizer and reward key."""
        super().__init__()
        self.buf  = replay_buffer
        self.tok  = tokenizer
        self.key  = accuracy_key
        self.thr  = threshold
        print("[ReplayBufferCallback] registered ✔️", flush=True)

    # ←–––– this fires AFTER loss.backward() and BEFORE scheduler/step().
    # It always receives both `inputs` and `outputs`.
    def on_train_batch_end(self, args, _state, _control, **batch_kwargs):
        """Inspect training outputs and enqueue prompts crossing the threshold."""
        outputs = batch_kwargs["outputs"]             # dict from training_step
        rewards = outputs.get("rewards", {})
        if self.key not in rewards:
            return

        batch_inputs = batch_kwargs["inputs"]
        input_ids = batch_inputs["input_ids"]
        accuracies = rewards[self.key].detach().cpu().tolist()

        added = sum(
            self._maybe_store_prompt(accuracy, token_ids)
            for accuracy, token_ids in zip(accuracies, input_ids)
        )

        self._log_batch_stats(
            args,
            batch_inputs.get("is_replay"),
            len(input_ids),
            added,
        )

    def _maybe_store_prompt(self, accuracy: float, token_ids) -> int:
        """Decode and store prompts meeting the accuracy threshold."""
        if accuracy < self.thr:
            return 0
        prompt = self.tok.decode(token_ids, skip_special_tokens=True)
        self.buf.add(prompt)
        return 1

    def _log_batch_stats(self, args, replay_flags, batch_size: int, added: int) -> None:
        """Log replay buffer statistics for debugging."""
        local_rank = getattr(args, "local_rank", -1)
        rank = local_rank if local_rank != -1 else 0
        replay_count = int(replay_flags.sum().item()) if replay_flags is not None else 0

        print(
            f"[ReplayBufferCallback][rank{rank}] added {added} new • "
            f"{replay_count}/{batch_size} replay • buffer = {len(self.buf)}",
            flush=True,
        )

#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Trainer callbacks that integrate Open-R1 runs with auxiliary services.

The callbacks defined here extend Hugging Face ``Trainer`` runs with behaviour
used by the Open-R1 experiments, focusing on pushing checkpoints to the Hub.
The functions are written to be picked up by Sphinx autodoc so they show up in
the generated API reference.
"""
from __future__ import annotations

import logging
import subprocess
from concurrent.futures import Future
from types import SimpleNamespace
try:  # pragma: no cover - optional dependency
    from transformers import (  # pylint: disable=import-error
        TrainerCallback,
        TrainerControl,
        TrainerState,
        TrainingArguments,
    )
except ImportError as import_error:  # pragma: no cover
    class TrainerCallback:  # type: ignore[too-few-public-methods]  # pylint: disable=too-few-public-methods
        """Fallback that surfaces a helpful error when Transformers is missing."""

        def __init__(self, *_args, **_kwargs) -> None:
            raise ImportError(
                "transformers is required to use Open-R1 Trainer callbacks. "
                "Install it with `pip install transformers`."
            ) from import_error

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
    """Return ``True`` when the ``sinfo`` binary is available (SLURM).

    :returns: ``True`` if ``sinfo`` can be executed successfully, ``False`` otherwise.
    """

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

class PushToHubRevisionCallback(TrainerCallback):  # pylint: disable=too-few-public-methods
    """Callback that pushes checkpoints to the Hub using revision tags."""

    def __init__(self, model_cfg):
        """Store the model configuration used when scheduling pushes.

        :param model_cfg: Transformer model configuration namespace.
        :returns: ``None``. Initialises logger handles for later use.
        """
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
        """Push the current checkpoint to the Hub and optionally schedule benchmarks.

        :param args: Training arguments describing Hub targets and output dirs.
        :param state: Trainer state containing the current global step.
        :param _control: Trainer control object (unused).
        :param _kwargs: Additional arguments ignored by this hook.
        :returns: ``None``. Enqueues push jobs and optional follow-up benchmarks.
        """
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
                try:
                    from .evaluation import run_benchmark_jobs  # pylint: disable=import-outside-toplevel
                except ImportError as exc:  # pragma: no cover - optional dependency
                    self.log.warning("Upload finished but evaluation helpers missing: %s", exc)
                    return
                self.log.info("Upload done – submitting benchmark job.")
                dummy.benchmarks = args.benchmarks
                run_benchmark_jobs(dummy, self.model_cfg)
            fut.add_done_callback(_after)  # pylint: disable=no-member

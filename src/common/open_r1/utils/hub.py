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

"""Helpers for pushing training artifacts to the Hugging Face Hub."""

from __future__ import annotations

import importlib
import logging
import re
from concurrent.futures import Future
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:  # pragma: no cover - typing only
    from trl import GRPOConfig, SFTConfig
else:  # pragma: no cover - runtime fallback
    GRPOConfig = SFTConfig = Any


logger = logging.getLogger(__name__)

_huggingface_cache: Dict[str, Any] = {"error": None, "helpers": None}
_transformers_cache: Dict[str, Any] = {"error": None, "module": None}


def _require_huggingface_hub():
    """Return huggingface_hub helpers or raise a clear installation error."""
    helpers = _huggingface_cache.get("helpers")
    if helpers is not None:
        return helpers

    if _huggingface_cache.get("error") is not None:
        raise ImportError(
            "huggingface_hub is required for Hub interactions. "
            "Install it with `pip install huggingface_hub`."
        ) from _huggingface_cache["error"]

    try:
        hub_module = importlib.import_module("huggingface_hub")
    except ImportError as exc:  # pragma: no cover - optional dependency
        _huggingface_cache["error"] = exc
        raise ImportError(
            "huggingface_hub is required for Hub interactions. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    helpers = (
        hub_module.create_repo,
        hub_module.list_repo_commits,
        hub_module.create_branch,
        hub_module.upload_folder,
        hub_module.repo_exists,
        hub_module.list_repo_refs,
        hub_module.list_repo_files,
        hub_module.get_safetensors_metadata,
    )
    _huggingface_cache["helpers"] = helpers
    return helpers


def _require_transformers():
    """Return AutoConfig or raise a clear installation error."""
    module = _transformers_cache.get("module")
    if module is not None:
        return module.AutoConfig

    if _transformers_cache.get("error") is not None:
        raise ImportError(
            "transformers is required for reading model configurations. "
            "Install it with `pip install transformers`."
        ) from _transformers_cache["error"]

    try:
        module = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - optional dependency
        _transformers_cache["error"] = exc
        raise ImportError(
            "transformers is required for reading model configurations. "
            "Install it with `pip install transformers`."
        ) from exc

    _transformers_cache["module"] = module
    return module.AutoConfig


def push_to_hub_revision(
    training_args: SFTConfig | GRPOConfig,
    extra_ignore_patterns=None,
) -> Future:
    """Push the model checkpoint to a dedicated branch on the Hugging Face Hub.

    :param training_args: Training arguments with Hub credentials and metadata.
    :param extra_ignore_patterns: Additional glob patterns excluded from the upload.
    :returns: Future representing the asynchronous upload operation.
    """

    (
        create_repo_fn,
        list_repo_commits_fn,
        create_branch_fn,
        upload_folder_fn,
        *_,
    ) = _require_huggingface_hub()

    if extra_ignore_patterns is None:
        extra_ignore_patterns = []

    # Create a repo if it doesn't exist yet
    repo_url = create_repo_fn(
        repo_id=training_args.hub_model_id,
        private=True,
        exist_ok=True,
    )
    # Get initial commit to branch from
    initial_commit = list_repo_commits_fn(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch_fn(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info("Created target repo at %s", repo_url)
    logger.info("Pushing to the Hub revision %s...", training_args.hub_model_revision)
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder_fn(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(
        "Pushed to %s revision %s successfully!",
        repo_url,
        training_args.hub_model_revision,
    )

    return future


def get_param_count_from_repo_id(repo_id: str) -> int:
    """Return parameter count from safetensors metadata or repo naming pattern.

    The repo ID can include human-readable counts such as 42m, 1.5b, 0.5m, or
    products like 8x7b. If metadata is unavailable, those tokens are parsed to
    estimate the parameter count.

    :param repo_id: Hugging Face repository identifier.
    :returns: Approximate parameter count, or ``-1`` when it cannot be determined.
    """
    (
        *_,
        get_safetensors_metadata_fn,
    ) = _require_huggingface_hub()

    try:
        metadata = get_safetensors_metadata_fn(repo_id)
        return list(metadata.parameter_count.values())[0]
    except Exception:  # pylint: disable=broad-exception-caught
        # Pattern to match products (like 8x7b) and single values (like 42m)
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for _, number1, _, _, number2, _, unit in matches:
            if number2:  # If there's a second number, it's a product
                number = float(number1) * float(number2)
            else:  # Otherwise, it's a single value
                number = float(number1)

            if unit == "b":
                number *= 1_000_000_000  # Convert to billion
            elif unit == "m":
                number *= 1_000_000  # Convert to million

            param_counts.append(number)

        if param_counts:
            return int(max(param_counts))
        return -1


def get_gpu_count_for_vllm(model_name: str, revision: str = "main", num_gpus: int = 8) -> int:
    """Return a GPU count compatible with vLLM attention head constraints.

    :param model_name: Model repository identifier.
    :param revision: Repository revision to inspect.
    :param num_gpus: Maximum number of GPUs to attempt.
    :returns: Number of GPUs that satisfy head divisibility constraints.
    """
    auto_config_cls = _require_transformers()
    config = auto_config_cls.from_pretrained(
        model_name,
        revision=revision,
        trust_remote_code=True,
    )
    # Get number of attention heads
    num_heads = config.num_attention_heads
    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(
            "Reducing num_gpus from %s to %s to make num_heads divisible by num_gpus",
            num_gpus,
            num_gpus - 1,
        )
        num_gpus -= 1
    return num_gpus

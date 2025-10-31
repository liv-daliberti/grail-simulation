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

"""Configuration helpers for open_r1 training scripts."""

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

try:  # pragma: no cover - optional dependency
    import trl
except ImportError:  # pragma: no cover - optional dependency
    trl = None  # type: ignore[assignment]

ScriptArgumentsBase = trl.ScriptArguments if trl is not None else object
GRPOConfigBase = trl.GRPOConfig if trl is not None else object
SFTConfigBase = trl.SFTConfig if trl is not None else object


@dataclass
class DatasetConfig:
    """Configuration for a dataset in a mixture."""

    id: str  # pylint: disable=invalid-name
    config: Optional[str] = None
    split: str = "train"
    columns: Optional[list[str]] = None
    weight: Optional[float] = None


@dataclass
class DatasetMixtureConfig:
    """Configuration for a mixture of datasets."""

    datasets: list[DatasetConfig]
    seed: int = 0
    test_split_size: Optional[float] = None


@dataclass
class ScriptArguments(ScriptArgumentsBase):
    """
    Extended version of ScriptArguments with support for dataset mixtures.

    :param dataset_mixture: Configuration for creating dataset mixtures with advanced
        options. Example::

            dataset_mixture:
              datasets:
                - id: dataset_id1
                  config: config_name
                  columns:
                    - col1
                    - col2
                  weight: 0.5
                - id: dataset_id2
                  config: config_name
                  columns:
                    - col1
                    - col2
                  weight: 0.5
              seed: 42
              test_split_size: 0.1
    :type dataset_mixture: dict[str, Any] | None
    """

    # Override the dataset_name to make it optional
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "Dataset name. Can be omitted if using dataset_mixture."}
    )
    dataset_mixture: Optional[dict[str, Any]] = field(
        default=None,
        metadata={
            "help": (
                "Configuration for creating dataset mixtures with advanced options "
                "like shuffling."
            )
        },
    )

    def __post_init__(self):
        """Normalize dataset mixture configuration after initialisation.

        Ensures at least one of ``dataset_name`` or ``dataset_mixture`` is provided,
        converts the mixture dictionary into :class:`DatasetMixtureConfig`, and
        validates column consistency across components.
        """

        if trl is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "trl must be installed to use ScriptArguments "
                "(pip install trl)."
            )

        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]

        if self.dataset_name is None and self.dataset_mixture is None:
            raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

        if self.dataset_mixture is not None:
            if not isinstance(self.dataset_mixture, dict) or "datasets" not in self.dataset_mixture:
                raise ValueError(
                    "dataset_mixture must be a dictionary with a 'datasets' key. "
                    "Expected format: {'datasets': [...], 'seed': int}"
                )

            datasets_list = []
            datasets_data = self.dataset_mixture.get("datasets", [])

            if isinstance(datasets_data, list):
                for dataset_config in datasets_data:
                    datasets_list.append(
                        DatasetConfig(
                            id=dataset_config.get("id"),
                            config=dataset_config.get("config"),
                            split=dataset_config.get("split", "train"),
                            columns=dataset_config.get("columns"),
                            weight=dataset_config.get("weight", 1.0),
                        )
                    )
            else:
                raise ValueError("'datasets' must be a list of dataset configurations")

            self.dataset_mixture = DatasetMixtureConfig(
                datasets=datasets_list,
                seed=self.dataset_mixture.get("seed", 0),
                test_split_size=self.dataset_mixture.get("test_split_size", None),
            )

            # Check that column names are consistent across all dataset configs
            columns_sets = [
                set(dataset.columns)
                for dataset in datasets_list
                if dataset.columns is not None
            ]
            if columns_sets:
                first_columns = columns_sets[0]
                if not all(columns == first_columns for columns in columns_sets):
                    raise ValueError(
                        "Column names must be consistent across all dataset configurations "
                        "in a mixture. "
                        f"Found different column sets: {[list(cols) for cols in columns_sets]}"
                    )


# NOTE: Consider adding shared options with a mixin to reduce code duplication.
@dataclass
class GRPOConfig(GRPOConfigBase):  # pylint: disable=too-many-instance-attributes
    """Arguments for callbacks, benchmarks, and related settings."""

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": (
                "The callbacks to run during training. Recognised values currently "
                "include 'push_to_hub_revision' to enable automatic checkpoint syncing."
            )
        },
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the Hub revision."},
    )
    push_to_hub_revision: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to push checkpoints to a unique Hub revision whenever the trainer "
                "saves. Requires `hub_model_id` and (optionally) honours `hub_model_revision`."
            )
        },
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use."},
    )

    def __post_init__(self) -> None:
        """Validate optional dependencies before continuing configuration.

        :returns: ``None``. Raises informative errors when dependencies are missing.
        """

        if trl is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "trl must be installed to construct GRPOConfig "
                "(pip install trl)."
            )
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]

@dataclass
class SFTConfig(SFTConfigBase):
    """Arguments for callbacks, benchmarks, and related settings."""

    # pylint: disable=too-many-instance-attributes

    benchmarks: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "The benchmarks to run after training."},
    )
    callbacks: list[str] = field(
        default_factory=lambda: [],
        metadata={
            "help": (
                "The callbacks to run during training. Recognised values currently "
                "include 'push_to_hub_revision' to enable automatic checkpoint syncing."
            )
        },
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "The chat template to use."},
    )
    system_prompt: Optional[str] = field(
        default=None,
        metadata={"help": "The optional system prompt to use for benchmarking."},
    )
    hub_model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "The Hub model branch to push the model to."},
    )
    overwrite_hub_revision: bool = field(
        default=False,
        metadata={"help": "Whether to overwrite the Hub revision."},
    )
    push_to_hub_revision: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to push checkpoints to a unique Hub revision whenever the trainer "
                "saves. Requires `hub_model_id` and (optionally) honours `hub_model_revision`."
            )
        },
    )
    wandb_entity: Optional[str] = field(
        default=None,
        metadata={"help": "The entity to store runs under."},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "The project to store runs under."},
    )
    wandb_run_group: Optional[str] = field(
        default=None,
        metadata={"help": "The group to store runs under."},
    )

    def __post_init__(self) -> None:
        """Validate optional dependencies before continuing configuration.

        :returns: ``None``. Raises informative errors when dependencies are missing.
        """

        if trl is None:  # pragma: no cover - optional dependency guard
            raise ImportError(
                "trl must be installed to construct SFTConfig "
                "(pip install trl)."
            )
        if hasattr(super(), "__post_init__"):
            super().__post_init__()  # type: ignore[misc]


@dataclass
class GRPOScriptArguments(ScriptArguments):  # pylint: disable=too-many-instance-attributes
    """
    Script arguments for the GRPO training script.

    :param reward_funcs: List of reward functions. Possible values: ``"accuracy"``,
        ``"format"``, ``"reasoning_steps"``, ``"cosine"``, ``"repetition_penalty"``,
        ``"length"``, ``"tag_count"``, ``"code"``, ``"code_format"``,
        ``"soft_overlong_punishment"``.
    :type reward_funcs: list[str]
    :param cosine_min_value_wrong: Minimum reward for cosine scaling for wrong answers.
    :type cosine_min_value_wrong: float
    :param cosine_max_value_wrong: Maximum reward for cosine scaling for wrong answers.
    :type cosine_max_value_wrong: float
    :param cosine_min_value_correct: Minimum reward for cosine scaling for correct answers.
    :type cosine_min_value_correct: float
    :param cosine_max_value_correct: Maximum reward for cosine scaling for correct answers.
    :type cosine_max_value_correct: float
    :param cosine_max_len: Maximum length for cosine scaling.
    :type cosine_max_len: int
    :param code_language: Language for the code format reward.
    :type code_language: str
    :param max_completion_len: Maximum number of tokens in a completion.
    :type max_completion_len: int
    :param soft_punish_cache: Minimum number of tokens in a completion before the
        hard penalty applies.
    :type soft_punish_cache: int
    """

    # pylint: disable=too-many-instance-attributes
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "tag_count"],
        metadata={
            "help": (
                "List of reward functions. Possible values: 'accuracy', 'format', "
                "'reasoning_steps', 'cosine', 'repetition_penalty', 'length', "
                "'tag_count', 'code', 'code_format', 'soft_overlong_punishment'."
            )
        },
    )
    cosine_min_value_wrong: float = field(
        default=0.0,
        metadata={"help": "Minimum reward for wrong answers"},
    )
    cosine_max_value_wrong: float = field(
        default=-0.5,
        metadata={"help": "Maximum reward for wrong answers"},
    )
    cosine_min_value_correct: float = field(
        default=0.5,
        metadata={"help": "Minimum reward for correct answers"},
    )
    cosine_max_value_correct: float = field(
        default=1.0,
        metadata={"help": "Maximum reward for correct answers"},
    )
    cosine_max_len: int = field(
        default=1000,
        metadata={"help": "Maximum length for scaling"},
    )
    repetition_n_grams: int = field(
        default=3,
        metadata={"help": "Number of n-grams for repetition penalty reward"},
    )
    repetition_max_penalty: float = field(
        default=-1.0,
        metadata={
            "help": "Maximum (negative) penalty for the repetition penalty reward."
        },
    )
    code_language: str = field(
        default="python",
        # '(?:python|cpp)'
        metadata={
            "help": (
                "Language for code format reward. Choose any language with templates "
                "supported by the local format checker."
            ),
            "choices": ["python", "javascript", "r", "java", "bash", "cpp"],
        },
    )
    parallel_code_exec_per_proc: int = field(
        default=2,
        metadata={
            "help": (
                "Retained for compatibility; local code evaluation ignores this value."
            )
        },
    )

    max_completion_len: int = field(
        default=16384,
        metadata={"help": "Maximum number of characters in completion."},
    )
    soft_punish_cache: int = field(
        default=4096,
        metadata={"help": "Minimum number of characters in completion."},
    )

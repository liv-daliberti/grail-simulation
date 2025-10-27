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

"""Dataset loading and mixture utilities for open_r1 training workflows."""

import logging

try:  # pragma: no cover - optional dependency
    import datasets as _DATASETS  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _DATASETS = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from datasets import concatenate_datasets  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    concatenate_datasets = None  # type: ignore

from common.hf_datasets import DatasetDict

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def _require_datasets_module():
    """Return the optional datasets module or raise a helpful error."""

    if _DATASETS is None:
        raise ImportError(
            "The 'datasets' package is required to load training data. "
            "Install it with `pip install datasets`."
        )
    return _DATASETS


def _require_concatenate():
    """Return the concatenate_datasets helper or raise when unavailable."""

    if concatenate_datasets is None:
        raise ImportError(
            "concatenate_datasets is unavailable because the 'datasets' package is not installed. "
            "Install it with `pip install datasets`."
        )
    return concatenate_datasets


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    :param args: Script arguments containing dataset configuration.
    :type args: ScriptArguments
    :return: Loaded dataset dictionary.
    :rtype: DatasetDict
    """
    datasets_module = _require_datasets_module()

    if args.dataset_name and not args.dataset_mixture:
        logger.info("Loading dataset: %s", args.dataset_name)
        return datasets_module.load_dataset(args.dataset_name, args.dataset_config)
    if args.dataset_mixture:
        logger.info(
            "Creating dataset mixture with %s datasets",
            len(args.dataset_mixture.datasets),
        )
        seed = args.dataset_mixture.seed
        datasets_list = []

        for dataset_config in args.dataset_mixture.datasets:
            logger.info(
                "Loading dataset for mixture: %s (config: %s)",
                dataset_config.id,
                dataset_config.config,
            )
            dataset_slice = datasets_module.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                dataset_slice = dataset_slice.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                dataset_slice = dataset_slice.shuffle(seed=seed).select(
                    range(int(len(dataset_slice) * dataset_config.weight))
                )
                logger.info(
                    "Subsampled dataset '%s' (config: %s) with weight=%s to %s examples",
                    dataset_config.id,
                    dataset_config.config,
                    dataset_config.weight,
                    len(dataset_slice),
                )

            datasets_list.append(dataset_slice)

        if datasets_list:
            concat_fn = _require_concatenate()
            combined_dataset = concat_fn(datasets_list)
            combined_dataset = combined_dataset.shuffle(seed=seed)
            logger.info(
                "Created dataset mixture with %s examples",
                len(combined_dataset),
            )

            if args.dataset_mixture.test_split_size is not None:
                combined_dataset = combined_dataset.train_test_split(
                    test_size=args.dataset_mixture.test_split_size,
                    seed=seed,
                )
                logger.info(
                    "Split dataset into train and test sets with test size: %s",
                    args.dataset_mixture.test_split_size,
                )
                return combined_dataset
            return datasets_module.DatasetDict({"train": combined_dataset})
        raise ValueError("No datasets were loaded from the mixture configuration")

    raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

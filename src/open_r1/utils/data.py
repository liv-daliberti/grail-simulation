"""Helpers for loading datasets and mixtures used by the training scripts."""

import logging
from typing import TYPE_CHECKING, Any

try:  # pragma: no cover - optional dependency
    import datasets as _DATASETS  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    _DATASETS = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from datasets import concatenate_datasets  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    concatenate_datasets = None  # type: ignore

from ..configs import ScriptArguments

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import DatasetDict
else:  # pragma: no cover - runtime fallback
    DatasetDict = Any  # type: ignore


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
            ds = datasets_module.load_dataset(
                dataset_config.id,
                dataset_config.config,
                split=dataset_config.split,
            )
            if dataset_config.columns is not None:
                ds = ds.select_columns(dataset_config.columns)
            if dataset_config.weight is not None:
                ds = ds.shuffle(seed=seed).select(
                    range(int(len(ds) * dataset_config.weight))
                )
                logger.info(
                    "Subsampled dataset '%s' (config: %s) with weight=%s to %s examples",
                    dataset_config.id,
                    dataset_config.config,
                    dataset_config.weight,
                    len(ds),
                )

            datasets_list.append(ds)

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

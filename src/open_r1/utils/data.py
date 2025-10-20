import logging

import datasets
from datasets import DatasetDict, concatenate_datasets

from ..configs import ScriptArguments


logger = logging.getLogger(__name__)


def get_dataset(args: ScriptArguments) -> DatasetDict:
    """Load a dataset or a mixture of datasets based on the configuration.

    Args:
        args (ScriptArguments): Script arguments containing dataset configuration.

    Returns:
        DatasetDict: The loaded datasets.
    """
    if args.dataset_name and not args.dataset_mixture:
        logger.info("Loading dataset: %s", args.dataset_name)
        return datasets.load_dataset(args.dataset_name, args.dataset_config)
    elif args.dataset_mixture:
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
            ds = datasets.load_dataset(
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
            combined_dataset = concatenate_datasets(datasets_list)
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
            else:
                return DatasetDict({"train": combined_dataset})
        else:
            raise ValueError("No datasets were loaded from the mixture configuration")

    else:
        raise ValueError("Either `dataset_name` or `dataset_mixture` must be provided")

"""Utility for wiring Weights & Biases environment variables from configs."""

import os


def init_wandb_training(training_args):
    """Populate Weights & Biases environment variables from training arguments.

    :param training_args: Namespace or dataclass with ``wandb_*`` attributes.
    """
    if training_args.wandb_entity is not None:
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity
    if training_args.wandb_project is not None:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
    if training_args.wandb_run_group is not None:
        os.environ["WANDB_RUN_GROUP"] = training_args.wandb_run_group

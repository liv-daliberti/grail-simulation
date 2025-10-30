"""Tests for :mod:`open_r1.configs` dataset handling logic."""

from __future__ import annotations

import pytest

pytest.importorskip("trl", reason="trl must be installed to run configuration tests.")

from open_r1.configs import (  # pylint: disable=wrong-import-position
    DatasetConfig,
    DatasetMixtureConfig,
    ScriptArguments,
)


def test_script_arguments_requires_name_or_mixture() -> None:
    with pytest.raises(ValueError, match="Either `dataset_name` or `dataset_mixture` must be provided"):
        ScriptArguments(dataset_name=None, dataset_mixture=None)  # type: ignore[arg-type]


def test_dataset_mixture_requires_datasets_key() -> None:
    with pytest.raises(ValueError, match="dataset_mixture must be a dictionary"):
        ScriptArguments(dataset_mixture="not-a-dict")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="dataset_mixture must be a dictionary"):
        ScriptArguments(dataset_mixture={"seed": 42})  # type: ignore[arg-type]


def test_dataset_mixture_requires_list_of_configs() -> None:
    with pytest.raises(ValueError, match="'datasets' must be a list"):
        ScriptArguments(dataset_mixture={"datasets": "invalid"})  # type: ignore[arg-type]


def test_dataset_mixture_enforces_column_consistency() -> None:
    dataset_mixture = {
        "datasets": [
            {"id": "a", "columns": ["question", "answer"]},
            {"id": "b", "columns": ["question", "label"]},
        ]
    }

    with pytest.raises(ValueError, match="Column names must be consistent"):
        ScriptArguments(dataset_mixture=dataset_mixture)


def test_dataset_mixture_is_normalised_into_dataclasses() -> None:
    dataset_mixture = {
        "datasets": [
            {"id": "a", "config": "cfg", "weight": 0.25, "split": "validation"},
            {"id": "b"},  # Should fall back to defaults
        ],
        "seed": 123,
        "test_split_size": 0.1,
    }

    args = ScriptArguments(dataset_mixture=dataset_mixture)

    assert isinstance(args.dataset_mixture, DatasetMixtureConfig)
    assert args.dataset_mixture.seed == 123
    assert args.dataset_mixture.test_split_size == 0.1
    assert len(args.dataset_mixture.datasets) == 2

    first, second = args.dataset_mixture.datasets
    assert first == DatasetConfig(id="a", config="cfg", split="validation", columns=None, weight=0.25)
    assert second == DatasetConfig(id="b", config=None, split="train", columns=None, weight=1.0)

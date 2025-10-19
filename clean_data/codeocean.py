"""CodeOcean dataset helpers.

This module intentionally keeps a very small surface area and delegates the
heavy session-parsing logic to :mod:`clean_data.sessions`.  Historically the
implementation lived here; consolidating it in :mod:`clean_data.sessions`
eliminates duplicate code paths while keeping backwards-compatible entry
points under the ``clean_data.codeocean`` namespace.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import datasets
from datasets import DatasetDict

from clean_data.io import resolve_capsule_data_root as _resolve_capsule_data_root
from clean_data.sessions import build_codeocean_rows, split_dataframe

log = logging.getLogger("clean_grail")


def load_codeocean_dataset(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    """Build Hugging Face datasets from a CodeOcean capsule directory.

    :param dataset_name: Filesystem path to the capsule root or its ``data`` directory.
    :param validation_ratio: Proportion of rows assigned to the validation split.
    :returns: Dataset dictionary containing the constructed splits.
    :raises ValueError: If the capsule structure is missing or no usable rows exist.
    """

    root = Path(dataset_name).expanduser()
    data_root = _resolve_capsule_data_root(root)
    if data_root is None:
        raise ValueError(f"CodeOcean capsule data not found under {dataset_name}")

    log.info("Building dataset from CodeOcean capsule at %s", data_root)
    df = build_codeocean_rows(data_root)
    if df.empty:
        raise ValueError("No usable rows found in CodeOcean sessions")

    split_frames = split_dataframe(df, validation_ratio=validation_ratio)
    ds = {
        name: datasets.Dataset.from_pandas(frame, preserve_index=False)
        for name, frame in split_frames.items()
        if not frame.empty
    }
    log.info("CodeOcean rows: %s", {name: len(frame) for name, frame in split_frames.items()})
    return DatasetDict(ds)


def resolve_capsule_data_root(path: Path) -> Optional[Path]:
    """Expose the IO helper under the codeocean namespace for compatibility.

    :param path: Candidate path to the capsule root or the nested ``data`` directory.
    :returns: Normalized capsule root when detected, otherwise ``None``.
    """

    return _resolve_capsule_data_root(path)


__all__ = [
    "build_codeocean_rows",
    "load_codeocean_dataset",
    "resolve_capsule_data_root",
    "split_dataframe",
]

#!/usr/bin/env python3
"""Lightweight diagnostics for the clean-data pipeline source dataset.

This script mirrors the sanity checks we perform locally when the
GitHub Actions job starts failing because the raw dataset changed.
It loads the dataset referenced by ``GRAIL_SOURCE_DATASET`` (or the
``--dataset-name`` CLI argument), prints split-level row counts, and
reports whether the fields required by ``filter_prompt_ready`` are
present in the first available training example.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Iterable

try:
    from datasets import DatasetDict, load_dataset
except ImportError:  # pragma: no cover - handled in CI before running
    print("datasets package is required for diagnostics", file=sys.stderr)
    sys.exit(2)


GOLD_ID_CANDIDATES = (
    "target_video_id",
    "click_id",
    "clicked_id",
    "next_video_id",
    "next_video_raw_id",
    "label",
    "answer",
    "sol_id",
)


def _summarise_scalar(value: Any) -> str:
    """Return a short, human-readable description of ``value``."""

    if value is None:
        return "None"
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) > 120:
            stripped = f"{stripped[:117]}..."
        return f"str(len={len(value)}): {stripped or '(empty)'}"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
        try:
            length = len(value)  # type: ignore[arg-type]
        except TypeError:
            length = "?"
        preview = list(value[:2]) if isinstance(value, list) else []
        return f"{type(value).__name__}(len={length}, preview={preview})"
    return repr(value)


def _find_first_row(split) -> dict[str, Any] | None:  # type: ignore[override]
    """Return the first non-empty row from the provided split, if any."""

    if len(split) == 0:  # type: ignore[arg-type]
        return None
    row = split[0]
    if isinstance(row, dict):
        return row
    if hasattr(row, "to_pylist"):
        data = row.to_pylist()
        return data[0] if data else None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspect the dataset backing the clean-data pipeline."
    )
    parser.add_argument(
        "--dataset-name",
        help="dataset identifier to load; defaults to $GRAIL_SOURCE_DATASET",
    )
    parser.add_argument(
        "--print-row",
        action="store_true",
        help="include a JSON dump of the first training row (use sparingly).",
    )
    args = parser.parse_args(argv)

    dataset_name = args.dataset_name or os.environ.get("GRAIL_SOURCE_DATASET")
    if not dataset_name:
        print(
            "No dataset specified. Provide --dataset-name or set GRAIL_SOURCE_DATASET.",
            file=sys.stderr,
        )
        return 1

    print(f"[dataset] loading '{dataset_name}'")
    try:
        dataset = load_dataset(dataset_name)
    except Exception as err:  # noqa: BLE001
        print(f"[dataset] failed to load: {err}", file=sys.stderr)
        return 2

    if not isinstance(dataset, DatasetDict):
        print("[dataset] expected DatasetDict with splits", file=sys.stderr)
        return 3

    for split_name, split in dataset.items():
        print(f"[split] {split_name}: {len(split)} rows; columns={len(split.column_names)}")
    if "train" not in dataset:
        print("[warn] dataset has no 'train' split; nothing to inspect further.")
        return 0

    train_split = dataset["train"]
    if len(train_split) == 0:
        print("[warn] train split is empty.")
        return 0

    first_row = _find_first_row(train_split)
    if first_row is None:
        print("[warn] unable to read first train row.")
        return 0

    train_columns = set(train_split.column_names)
    print(f"[train] total columns: {len(train_columns)}")

    if "prompt" in train_columns:
        print(
            "[warn] 'prompt' column detected in raw data. "
            "This looks like the already-cleaned dataset; "
            "the pipeline expects raw session logs."
        )

    slate_value = first_row.get("slate_items_json")
    if slate_value is None:
        print("[error] first train row is missing 'slate_items_json'")
    else:
        summary = _summarise_scalar(slate_value)
        print(f"[train] slate_items_json -> {summary}")

    present_gold_fields = []
    for candidate in GOLD_ID_CANDIDATES:
        value = first_row.get(candidate)
        if value:
            present_gold_fields.append(candidate)
            summary = _summarise_scalar(value)
            print(f"[train] candidate gold field '{candidate}' -> {summary}")

    if not present_gold_fields:
        print(
            "[warn] none of the known gold-id fields are populated; "
            "set GRAIL_SOL_KEY to the correct column if it differs."
        )
    else:
        print(
            "[info] gold-id candidates present. "
            "Set GRAIL_SOL_KEY to one of: "
            + ", ".join(present_gold_fields)
        )

    if args.print_row:
        # Mask potentially sensitive content by truncating values.
        truncated = {
            key: (value[:200] + "...") if isinstance(value, str) and len(value) > 200 else value
            for key, value in first_row.items()
        }
        print("[train] first row sample:")
        print(json.dumps(truncated, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared dataset helpers for GRAIL / GRPO pipelines."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from datasets import DatasetDict
except ImportError:  # pragma: no cover - optional dependency
    DatasetDict = Any  # type: ignore


GoldLookup = Callable[[Dict[str, Any], Optional[str]], str]
SlateLoader = Callable[[Dict[str, Any]], Sequence[Dict[str, Any]]]
IndexResolver = Callable[[str, Sequence[Dict[str, Any]]], int]


def slate_has_gold(  # pylint: disable=too-many-arguments
    example: Dict[str, Any],
    solution_key: Optional[str],
    *,
    load_slate_items: SlateLoader,
    lookup_gold_id: GoldLookup,
    resolve_gold_index: IndexResolver,
    minimum_index: int = 1,
) -> bool:
    """Return ``True`` when ``example`` contains a slate and a matching gold answer."""

    items = load_slate_items(example)
    if not items:
        return False
    gold_id = lookup_gold_id(example, solution_key)
    if not gold_id:
        return False
    return resolve_gold_index(gold_id, items) >= minimum_index


def drop_marked_rows(dataset: DatasetDict, train_split: str) -> None:
    """Remove rows flagged with ``__drop__`` from every split in-place."""

    if "__drop__" not in dataset[train_split].column_names:
        return
    for split in list(dataset.keys()):
        mask = [not flag for flag in dataset[split]["__drop__"]]
        keep_indices = [idx for idx, keep in enumerate(mask) if keep]
        dataset[split] = dataset[split].select(keep_indices)


__all__ = ["drop_marked_rows", "slate_has_gold"]

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

"""Dataset helpers used by the GRPO evaluation pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import common.data.hf_datasets as _hf_datasets
from common.open_r1.example_utils import row_to_training_example

DOWNLOAD_CONFIG_CLS, LOAD_DATASET, LOAD_FROM_DISK = _hf_datasets.get_dataset_loaders()
LOGGER = logging.getLogger("grpo.dataset")


@dataclass(frozen=True)
class PreparedExample:
    """Container describing a single evaluation-ready GRPO prompt.

    :ivar list[dict[str, str]] messages: Conversation payload forwarded to the model.
    :ivar int gold_index: 1-based index of the correct option in ``messages``.
    :ivar str gold_id: Identifier corresponding to the ground-truth option.
    :ivar int n_options: Number of options appearing in the prompt slate.
    :ivar Mapping[str, Any] raw_row: Original dataset row used to build the prompt.
    """

    messages: list[dict[str, str]]
    gold_index: int
    gold_id: str
    n_options: int
    raw_row: Mapping[str, Any]

    @property
    def issue(self) -> str:
        """Return the issue label associated with the example.

        :returns: Normalised issue label (empty string when unknown).
        :rtype: str
        """

        return str(self.raw_row.get("issue") or "").strip()

    @property
    def participant_study(self) -> str:
        """Return the participant-study identifier for the example.

        :returns: Participant-study key used for downstream metrics.
        :rtype: str
        """

        return str(self.raw_row.get("participant_study") or "").strip()

    @property
    def position_index(self) -> int:
        """Expose the position index when present (else ``-1``).

        :returns: 0-based position index extracted from the raw row or ``-1``.
        :rtype: int
        """

        try:
            return int(self.raw_row.get("position_index") or -1)
        except (TypeError, ValueError):
            return -1


def _load_dataset_from_path(path: Path):
    """Load a Hugging Face dataset stored on disk.

    :param path: Filesystem path pointing to the dataset directory.
    :returns: Loaded dataset object as returned by ``datasets.load_from_disk``.
    """

    _hf_datasets.require_dataset_support(needs_local=True)
    if LOAD_FROM_DISK is None:  # pragma: no cover - defensive
        raise RuntimeError("datasets.load_from_disk is unavailable.")
    LOGGER.info("[DATASET] loading local dataset from %s", path)
    print(f"[grpo.dataset] loading local dataset from {path}", flush=True)
    return LOAD_FROM_DISK(str(path))


def _load_dataset_from_hub(
    dataset_id: str,
    *,
    cache_dir: str | None,
    revision: str | None = None,
):
    """Download a Hugging Face dataset from the hub.

    :param dataset_id: Hugging Face dataset identifier.
    :param cache_dir: Optional datasets cache directory.
    :param revision: Optional branch, tag, or commit identifier.
    :returns: Loaded dataset object as produced by ``datasets.load_dataset``.
    """

    _hf_datasets.require_dataset_support()
    if LOAD_DATASET is None or DOWNLOAD_CONFIG_CLS is None:  # pragma: no cover - defensive
        raise RuntimeError("datasets.load_dataset is unavailable.")
    download_config = DOWNLOAD_CONFIG_CLS(  # type: ignore[misc]
        resume_download=True,
        max_retries=2,
    )
    LOGGER.info(
        "[DATASET] fetching hub dataset id=%s revision=%s cache_dir=%s",
        dataset_id,
        revision or "default",
        cache_dir or "<default>",
    )
    revision_text = revision or "default"
    cache_text = cache_dir or "<default>"
    print(
        (
            f"[grpo.dataset] fetching hub dataset id={dataset_id} "
            f"revision={revision_text} cache_dir={cache_text}"
        ),
        flush=True,
    )
    return LOAD_DATASET(  # type: ignore[misc]
        dataset_id,
        cache_dir=cache_dir,
        download_config=download_config,
        revision=revision,
    )


def load_dataset_split(
    dataset_name: str,
    *,
    split: str,
    cache_dir: str | None = None,
) -> Sequence[Mapping[str, Any]]:
    """Return the materialised dataset split used during evaluation.

    :param dataset_name: Local path or Hugging Face dataset identifier.
    :param split: Preferred split label (fallback heuristics applied).
    :param cache_dir: Optional cache directory for Hub downloads.
    :returns: Sequence of raw dataset rows ready for preprocessing.
    :rtype: Sequence[Mapping[str, Any]]
    """

    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        dataset = _load_dataset_from_path(dataset_path)
        source_repr = f"disk:{dataset_path}"
    else:
        dataset = _load_dataset_from_hub(dataset_name, cache_dir=cache_dir)
        source_repr = f"hub:{dataset_name}"

    if hasattr(dataset, "keys"):
        for candidate in (split, "validation", "eval", "test"):
            if candidate in dataset:  # type: ignore[operator]
                current_split = dataset[candidate]  # type: ignore[index]
                rows = list(current_split)
                LOGGER.info(
                    "[DATASET] using split=%s from %s (rows=%d)",
                    candidate,
                    source_repr,
                    len(rows),
                )
                print(
                    (
                        f"[grpo.dataset] using split={candidate} "
                        f"from {source_repr} rows={len(rows)}"
                    ),
                    flush=True,
                )
                return rows
        raise RuntimeError(
            f"Unable to locate evaluation split '{split}' in dataset '{dataset_name}'."
        )
    if hasattr(dataset, "split"):
        rows = list(dataset)  # type: ignore[arg-type]
        LOGGER.info(
            "[DATASET] using iterable dataset from %s (rows=%d)", source_repr, len(rows)
        )
        print(
            f"[grpo.dataset] using iterable dataset from {source_repr} rows={len(rows)}",
            flush=True,
        )
        return rows
    rows = list(dataset)
    LOGGER.info("[DATASET] materialised dataset from %s (rows=%d)", source_repr, len(rows))
    print(
        f"[grpo.dataset] materialised dataset from {source_repr} rows={len(rows)}",
        flush=True,
    )
    return rows


def prepare_examples(
    rows: Iterable[Mapping[str, Any]],
    *,
    system_prompt: str,
    solution_key: str | None,
    max_history: int,
) -> Iterator[PreparedExample]:
    """Yield :class:`PreparedExample` instances generated from raw rows.

    :param rows: Iterable of raw dataset rows.
    :param system_prompt: System prompt injected into the chat template.
    :param solution_key: Optional column containing the gold identifier.
    :param max_history: Maximum number of history turns to include.
    :returns: Iterator over validated :class:`PreparedExample` objects.
    :rtype: Iterator[PreparedExample]
    """

    for row in rows:
        example = row_to_training_example(
            row,
            system_prompt=system_prompt,
            solution_key=solution_key,
            max_history=max_history,
        )
        if not example:
            continue
        messages = example.get("prompt")
        gold_index = example.get("gold_index")
        gold_id = example.get("gold_id")
        n_options = example.get("n_options")
        if not isinstance(messages, list) or not messages:
            continue
        try:
            prepared = PreparedExample(
                messages=list(messages),
                gold_index=int(gold_index or -1),
                gold_id=str(gold_id or ""),
                n_options=int(n_options or 0),
                raw_row=row,
            )
        except (TypeError, ValueError):
            continue
        yield prepared

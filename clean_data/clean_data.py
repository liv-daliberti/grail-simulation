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

"""Orchestrate loading, filtering, validation, persistence, and reporting for
``clean_data`` datasets.

All functionality is provided under the repository's Apache 2.0 license; see
LICENSE for terms and conditions.
"""

from __future__ import annotations

import csv
import importlib
import json
import logging
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except (ImportError, OSError, RuntimeError, ValueError):
    torch = None  # type: ignore[assignment]

from clean_data._datasets import (
    DatasetDict,
    DatasetGenerationCastError,
    Features,
    HFSequence,
    Value,
    datasets,
    ensure_datasets_imported as _ensure_datasets_imported,
    load_dataset_with_column_union as _load_dataset_with_column_union,
)
from clean_data.filters import compute_issue_counts, filter_prompt_ready
from clean_data.prompt.constants import REQUIRED_PROMPT_COLUMNS
from clean_data.prompting import row_to_example

def _ensure_torch_env_compat() -> None:
    """Patch an installed or stubbed torch module for HF datasets compatibility.

    Some test environments install extremely lightweight torch stubs that
    don't expose class-like ``Tensor`` or required submodules. The Hugging Face
    ``datasets`` pickler checks for ``torch`` and calls ``issubclass(..., torch.Tensor)``,
    which raises a ``TypeError`` if ``torch.Tensor`` is not a class. This helper
    ensures the minimal attributes exist when a ``torch`` module is present.
    """

    if torch is None:
        return

    # Guarantee class-like attributes expected by datasets' dill integration
    if not isinstance(getattr(torch, "Tensor", type), type):
        setattr(torch, "Tensor", type("Tensor", (), {}))
    if not hasattr(torch, "Generator"):
        setattr(torch, "Generator", type("Generator", (), {}))
    # Provide torch.nn.Module for isinstance checks if missing
    if not hasattr(torch, "nn"):
        nn_mod = types.ModuleType("torch.nn")
        nn_mod.Module = type("Module", (), {})  # type: ignore[attr-defined]
        torch.nn = nn_mod  # type: ignore[attr-defined]
        sys.modules.setdefault("torch.nn", nn_mod)
    else:
        sys.modules.setdefault("torch.nn", torch.nn)
    # Provide a distributed stub with is_available flag
    dist_mod = getattr(torch, "distributed", types.ModuleType("torch.distributed"))
    if not hasattr(dist_mod, "is_available"):
        dist_mod.is_available = lambda: False  # type: ignore[attr-defined]
    torch.distributed = dist_mod  # type: ignore[attr-defined]
    sys.modules.setdefault("torch.distributed", torch.distributed)


_ensure_torch_env_compat()

log = logging.getLogger("clean_grail")


@lru_cache(maxsize=1)
def _load_prompt_feature_report_builder() -> Optional[Callable[..., None]]:
    """Return the optional prompt statistics report builder.

    :returns: Callable used to generate prompt feature reports, or ``None`` when unavailable.
    """

    target_attr = "generate_prompt_feature_report"
    try:
        module = importlib.import_module("clean_data.prompt.cli")
    except ModuleNotFoundError:  # pragma: no cover
        try:
            module = importlib.import_module("prompt_stats")
        except ModuleNotFoundError:
            return None
    return getattr(module, target_attr, None)


def _try_load_codeocean_dataset(
    capsule_path: Path,
    validation_ratio: float,
) -> Optional[DatasetDict]:
    """Attempt to rebuild a dataset from the CodeOcean capsule layout.

    :param capsule_path: Path to the CodeOcean capsule directory on disk.
    :param validation_ratio: Fraction of the capsule data reserved for validation.
    :returns: Reconstructed dataset when the capsule utilities are present, otherwise ``None``.
    """

    try:
        codeocean_module = importlib.import_module("clean_data.codeocean")
    except ModuleNotFoundError:  # pragma: no cover
        return None

    try:
        return codeocean_module.load_codeocean_dataset(
            str(capsule_path),
            validation_ratio=validation_ratio,
        )
    except ValueError:
        return None


@dataclass(frozen=True)
class BuildOptions:
    """Configuration used when building cleaned prompt datasets.

    :param validation_ratio: Share of examples reserved for the validation split.
    :param train_split: Name of the split treated as training data when present.
    :param validation_split: Name of the split treated as validation data when present.
    :param system_prompt: Optional override applied to the prompt ``system`` role.
    :param sol_key: Alternate column containing the gold answer identifier.
    :param max_history: Number of historical interactions preserved in ``state_text``.
    :param num_proc: Optional worker count forwarded to Hugging Face dataset utilities.
    """

    validation_ratio: float = 0.1
    train_split: str = "train"
    validation_split: str = "validation"
    system_prompt: Optional[str] = None
    sol_key: Optional[str] = None
    max_history: int = 12
    num_proc: Optional[int] = None


def load_raw(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    """Load raw prompt data from disk or the Hugging Face Hub.

    :param dataset_name: Local filesystem path, CodeOcean capsule directory, or dataset id.
    :param validation_ratio: Validation share used when rebuilding from a capsule directory.
    :returns: Hugging Face ``DatasetDict`` containing the available splits.
    :raises ValueError: If the provided file path has an unsupported extension.
    """

    _ensure_datasets_imported()

    path = Path(dataset_name).expanduser()
    if path.exists():
        if path.is_dir():
            capsule_ds = _try_load_codeocean_dataset(path, validation_ratio)
            if capsule_ds is not None:
                return capsule_ds
            log.info("Loading dataset from disk: %s", path)
            loaded_dataset = datasets.load_from_disk(str(path))
            if isinstance(loaded_dataset, DatasetDict):
                return loaded_dataset
            return DatasetDict({"train": loaded_dataset})
        if path.is_file():
            return _load_local_file_dataset(path)
    log.info("Loading dataset from hub: %s", dataset_name)
    try:
        loaded = datasets.load_dataset(dataset_name)
    except DatasetGenerationCastError as err:
        log.warning(
            "Falling back to column-union loader for dataset=%s due to schema mismatch (%s)",
            dataset_name,
            err,
        )
        return _load_dataset_with_column_union(dataset_name)
    return DatasetDict(dict(loaded.items()))


def _load_local_file_dataset(path: Path) -> DatasetDict:
    """Load a local dataset file, falling back to manual parsing when needed."""

    ext = path.suffix.lower()
    if ext in {".json", ".jsonl"}:
        try:
            loaded = datasets.load_dataset("json", data_files=str(path))
        except RuntimeError:
            return _load_json_fallback(path, lines=ext == ".jsonl")
        return DatasetDict(dict(loaded.items()))
    if ext in {".csv", ".tsv"}:
        try:
            loaded = datasets.load_dataset(
                "csv",
                data_files=str(path),
                delimiter="," if ext == ".csv" else "\t",
            )
        except RuntimeError:
            return _load_delimited_fallback(path, delimiter="," if ext == ".csv" else "\t")
        return DatasetDict(dict(loaded.items()))
    raise ValueError(f"Unsupported file type: {path}")


def _build_dataset(rows: list[dict[str, object]]) -> DatasetDict:
    dataset = datasets.Dataset.from_list(rows)
    return DatasetDict({"train": dataset})


def _normalize_rows(raw_rows: list[object]) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for entry in raw_rows:
        if isinstance(entry, dict):
            normalized.append(entry)
        else:
            normalized.append({"value": entry})
    return normalized


def _load_json_fallback(path: Path, *, lines: bool) -> DatasetDict:
    if lines:
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return _build_dataset(_normalize_rows(rows))
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if isinstance(payload, dict) and all(isinstance(value, list) for value in payload.values()):
        mapping = {
            split: datasets.Dataset.from_list(_normalize_rows(rows))
            for split, rows in payload.items()
        }
        return DatasetDict(mapping)
    if isinstance(payload, list):
        return _build_dataset(_normalize_rows(payload))
    raise ValueError(f"Unsupported JSON structure in dataset file: {path}")


def _load_delimited_fallback(path: Path, *, delimiter: str) -> DatasetDict:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        rows = [dict(row) for row in reader]
    return _build_dataset(_normalize_rows(rows))



def map_rows_to_examples(
    dataset: DatasetDict,
    *,
    system_prompt: Optional[str],
    sol_key: Optional[str],
    max_history: int,
    num_proc: Optional[int] = None,
) -> DatasetDict:
    """Convert interaction rows into cleaned prompt examples.

    :param dataset: Source datasets keyed by split.
    :param system_prompt: Optional system prompt override applied to every row.
    :param sol_key: Alternate column containing the gold next-video identifier.
    :param max_history: Maximum number of prior interactions to embed in ``state_text``.
    :param num_proc: Optional number of worker processes passed to ``datasets.Dataset.map``.
    :returns: Dataset mapping where each split has been converted to prompt-ready rows.
    """

    if num_proc is not None and num_proc < 1:
        raise ValueError("num_proc must be >= 1 when provided.")

    mapped = DatasetDict()
    for split_name, split_ds in dataset.items():
        map_kwargs = {
            "remove_columns": split_ds.column_names,
            "load_from_cache_file": False,
        }
        if num_proc is not None:
            map_kwargs["num_proc"] = num_proc
        mapped_split = split_ds.map(
            lambda ex, prompt=system_prompt: row_to_example(
                ex,
                prompt,
                sol_key,
                max_history,
            ),
            **map_kwargs,
        )
        mapped[split_name] = mapped_split
    return mapped


def ensure_shared_schema(datasets_map: Dict[str, datasets.Dataset]) -> Dict[str, datasets.Dataset]:
    """Align feature schemas across splits so they expose identical columns.

    :param datasets_map: Mapping of split name to dataset requiring schema alignment.
    :returns: New mapping with each split cast to the union of all features.
    """

    _ensure_datasets_imported()

    all_columns: set[str] = set()
    feature_template: Dict[str, Value | HFSequence] = {}
    for split_ds in datasets_map.values():
        for name, feature in split_ds.features.items():
            all_columns.add(name)
            feature_template.setdefault(name, feature)

    def _default_for_feature(feature: Value | HFSequence, length: int) -> list:
        """Return a filler column matching the provided feature spec.

        :param feature: Hugging Face feature describing the target column.
        :param length: Desired column length.
        :returns: List of placeholder values respecting the feature type.
        """

        if isinstance(feature, HFSequence):
            return [[] for _ in range(length)]
        return [None] * length

    merged_features = Features(feature_template)
    aligned: Dict[str, datasets.Dataset] = {}
    for split_name, split_ds in datasets_map.items():
        missing = [col for col in all_columns if col not in split_ds.column_names]
        if missing:
            for column in missing:
                feature = feature_template.get(column)
                filler = _default_for_feature(feature, len(split_ds))
                split_ds = split_ds.add_column(column, filler)
        split_ds = split_ds.cast(merged_features)
        aligned[split_name] = split_ds
    return aligned


def validate_required_columns(dataset: DatasetDict) -> None:
    """Validate that every split exposes the GRPO-required prompt columns.

    :param dataset: Dataset to inspect.
    :raises ValueError: If any required column is missing from a split.
    """

    for split_name, split_ds in dataset.items():
        missing = sorted(REQUIRED_PROMPT_COLUMNS - set(split_ds.column_names))
        if missing:
            raise ValueError(
                f"Split '{split_name}' is missing required columns for GRPO: {missing}"
            )


def build_clean_dataset(
    dataset_name: str,
    *,
    options: Optional[BuildOptions] = None,
) -> DatasetDict:
    """Produce cleaned prompt examples from a raw dataset source.

    :param dataset_name: Path or dataset identifier understood by :func:`load_raw`.
    :param options: Optional overrides controlling split names and prompt construction.
    :returns: Cleaned dataset containing ``train`` and optional ``validation`` splits.
    """

    opts = options or BuildOptions()

    _ensure_datasets_imported()

    raw = load_raw(dataset_name, validation_ratio=opts.validation_ratio)

    filtered = filter_prompt_ready(raw, sol_key=opts.sol_key, num_proc=opts.num_proc)
    issue_counts = compute_issue_counts(filtered)
    if issue_counts:
        log.info("Issue distribution per split: %s", issue_counts)

    mapped = map_rows_to_examples(
        filtered,
        system_prompt=opts.system_prompt,
        sol_key=opts.sol_key,
        max_history=opts.max_history,
        num_proc=opts.num_proc,
    )

    desired: Dict[str, datasets.Dataset] = {}
    if opts.train_split in mapped:
        desired["train"] = mapped[opts.train_split]
    else:
        first_split = next(iter(mapped.keys()))
        desired["train"] = mapped[first_split]

    if opts.validation_split in mapped and opts.validation_split != opts.train_split:
        desired["validation"] = mapped[opts.validation_split]

    final = DatasetDict(ensure_shared_schema(desired))
    validate_required_columns(final)
    return final


def dedupe_by_participant_issue(dataset: DatasetDict) -> DatasetDict:
    """Drop duplicate rows sharing the same participant and issue.

    The cleaning pipeline now retains every interaction row long enough to
    run prompt diagnostics (e.g., prior-history coverage). Downstream
    consumers that require the historical one-row-per-participant-and-issue
    view can apply this helper to recover the previous behavior.

    :param dataset: Dataset containing potential participant/issue duplicates.
    :returns: Dataset with duplicates removed within each split.
    """

    _ensure_datasets_imported()

    deduped_splits: Dict[str, datasets.Dataset] = {}
    for split_name, split_ds in dataset.items():
        if not split_ds:
            deduped_splits[split_name] = split_ds
            continue

        required_columns = {"participant_id", "issue"}
        if not required_columns.issubset(split_ds.column_names):
            deduped_splits[split_name] = split_ds
            continue

        frame = split_ds.to_pandas()
        if frame.empty:
            deduped_splits[split_name] = split_ds
            continue

        participant = frame["participant_id"].fillna("").astype(str).str.strip()
        issue = frame["issue"].fillna("").astype(str).str.strip()
        keys = participant + "||" + issue
        keep_mask = ~keys.duplicated()
        deduped_frame = frame.loc[keep_mask].copy()

        removed = len(frame) - len(deduped_frame)
        if removed:
            log.info(
                "Removed %d duplicate participant/issue rows from split '%s' (kept %d of %d).",
                removed,
                split_name,
                len(deduped_frame),
                len(frame),
            )

        from_pandas = getattr(datasets.Dataset, "from_pandas", None)
        if callable(from_pandas):
            deduped_dataset = from_pandas(
                deduped_frame,
                preserve_index=False,
            )
            if hasattr(deduped_dataset, "cast"):
                deduped_dataset = deduped_dataset.cast(split_ds.features)
            deduped_splits[split_name] = deduped_dataset
        else:
            frame_data = {
                column: deduped_frame[column].tolist() for column in deduped_frame.columns
            }
            deduped_dataset = datasets.Dataset.from_dict(frame_data)
            if hasattr(deduped_dataset, "cast"):
                deduped_dataset = deduped_dataset.cast(split_ds.features)
            deduped_splits[split_name] = deduped_dataset

    return DatasetDict(deduped_splits)


def save_dataset(dataset: DatasetDict, output_dir: Path | str) -> None:
    """Persist a cleaned dataset to disk using ``Dataset.save_to_disk``.

    :param dataset: Dataset to save.
    :param output_dir: Destination directory that will be created if necessary.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    log.info("Saved cleaned dataset to %s", output_path)


def generate_prompt_stats(
    dataset: DatasetDict,
    output_dir: Path | str,
    *,
    train_split: str = "train",
    validation_split: str = "validation",
) -> None:
    """Render summary statistics and figures for a cleaned dataset.

    :param dataset: Dataset containing cleaned prompt rows.
    :param output_dir: Directory where plots and Markdown summaries will be written.
    :param train_split: Name of the training split in ``dataset``.
    :param validation_split: Name of the validation split in ``dataset``.
    :raises ImportError: If the prompt reporting extras are not installed.
    """

    stats_path = Path(output_dir)
    stats_path.mkdir(parents=True, exist_ok=True)
    report_builder = _load_prompt_feature_report_builder()
    if report_builder is None:  # pragma: no cover
        raise ImportError(
            "Prompt statistics module not available; install clean_data[prompt] extras."
        ) from None
    report_builder(
        dataset,
        output_dir=stats_path,
        train_split=train_split,
        validation_split=validation_split,
    )


def export_issue_datasets(
    dataset: DatasetDict,
    output_dir: Path | str,
    issue_repo_map: Dict[str, str],
    *,
    push_to_hub: bool = False,
    hub_token: Optional[str] = None,
) -> None:
    """Write per-issue dataset subsets (and optionally push them to the hub).

    :param dataset: Cleaned dataset keyed by split.
    :param output_dir: Base directory where issue-specific datasets will be written.
    :param issue_repo_map: Mapping from issue name to Hugging Face repository id.
    :param push_to_hub: When ``True``, push each subset to its configured repository.
    :param hub_token: Authentication token used for Hugging Face uploads.
    """

    _ensure_datasets_imported()

    if not issue_repo_map and not push_to_hub:
        return

    if not all("issue" in split.column_names for split in dataset.values()):
        log.warning(
            "Issue-level exports requested, but 'issue' column missing in dataset"
        )
        return

    issues_in_data: set[str] = set(issue_repo_map.keys())
    for split_ds in dataset.values():
        if "issue" in split_ds.column_names:
            issues_in_data.update(split_ds.unique("issue"))

    base_dir = Path(output_dir)
    for issue_name in sorted(issues_in_data):
        if not issue_name:
            continue
        issue_ds = DatasetDict()
        for split_name, split_ds in dataset.items():
            subset = split_ds.filter(lambda row, name=issue_name: row.get("issue") == name)
            if len(subset):
                issue_ds[split_name] = subset
        if not issue_ds:
            log.warning("No rows for issue %s; skipping", issue_name)
            continue

        issue_dir = base_dir / issue_name
        issue_dir.mkdir(parents=True, exist_ok=True)
        log.info("Saving issue '%s' dataset to %s", issue_name, issue_dir)
        issue_ds.save_to_disk(str(issue_dir))

        repo_id = issue_repo_map.get(issue_name)
        if push_to_hub and repo_id:
            log.info("Pushing issue '%s' dataset to %s", issue_name, repo_id)
            issue_ds.push_to_hub(repo_id, token=hub_token)
            log.info(
                "Issue '%s' dataset successfully pushed to Hugging Face hub repository %s",
                issue_name,
                repo_id,
            )
    if push_to_hub:
        log.info("All requested issue datasets have been pushed to the Hugging Face hub.")


def parse_issue_repo_specs(specs: Optional[list[str]]) -> Dict[str, str]:
    """Parse CLI ``issue=repo`` assignments into a mapping.

    :param specs: Iterable of ``issue=repo`` strings provided on the command line.
    :returns: Mapping of normalized issue name to repository identifier.
    :raises ValueError: If any entry is malformed.
    """

    mapping: Dict[str, str] = {}
    if not specs:
        return mapping
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --issue-repo format: {spec!r}; expected issue=repo")
        issue, repo = spec.split("=", 1)
        mapping[issue.strip()] = repo.strip()
    return mapping

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    import datasets
    from datasets import Dataset, DatasetDict, Features, Sequence as HFSequence, Value
    from datasets.builder import DatasetGenerationCastError
except ImportError:  # pragma: no cover - optional dependency for linting
    datasets = None  # type: ignore
    Dataset = Any  # type: ignore
    DatasetDict = Any  # type: ignore
    Features = Any  # type: ignore
    HFSequence = Any  # type: ignore
    Value = Any  # type: ignore
    DatasetGenerationCastError = Exception  # type: ignore

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency for linting
    pd = None  # type: ignore

try:
    from fsspec.core import url_to_fs
except ImportError:  # pragma: no cover - optional dependency for linting
    url_to_fs = None  # type: ignore

from clean_data.filters import compute_issue_counts, filter_prompt_ready
from clean_data.prompt.constants import REQUIRED_PROMPT_COLUMNS
from clean_data.prompting import row_to_example

log = logging.getLogger("clean_grail")


@lru_cache(maxsize=1)
def _load_prompt_feature_report_builder() -> Optional[Callable[..., None]]:
    """Return the optional prompt statistics report builder."""

    try:
        from clean_data.prompt.cli import (  # type: ignore  # pylint: disable=import-outside-toplevel
            generate_prompt_feature_report as builder,
        )
        return builder
    except ModuleNotFoundError:  # pragma: no cover
        try:
            from prompt_stats import generate_prompt_feature_report as builder  # type: ignore  # pylint: disable=import-outside-toplevel
            return builder
        except ModuleNotFoundError:  # pragma: no cover
            return None


def _try_load_codeocean_dataset(
    capsule_path: Path,
    validation_ratio: float,
) -> Optional[DatasetDict]:
    """Attempt to rebuild a dataset from the CodeOcean capsule layout."""

    try:
        from clean_data import codeocean as codeocean_module  # type: ignore  # pylint: disable=import-outside-toplevel
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
    """Configuration used when building cleaned prompt datasets."""

    validation_ratio: float = 0.1
    train_split: str = "train"
    validation_split: str = "validation"
    system_prompt: Optional[str] = None
    sol_key: Optional[str] = None
    max_history: int = 12


def load_raw(dataset_name: str, validation_ratio: float = 0.1) -> DatasetDict:
    """Load raw prompt data from disk or the Hugging Face Hub.

    :param dataset_name: Local filesystem path, CodeOcean capsule directory, or dataset id.
    :param validation_ratio: Validation share used when rebuilding from a capsule directory.
    :returns: Hugging Face ``DatasetDict`` containing the available splits.
    :raises ValueError: If the provided file path has an unsupported extension.
    """

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
            ext = path.suffix.lower()
            if ext in {".json", ".jsonl"}:
                loaded = datasets.load_dataset("json", data_files=str(path))
            elif ext in {".csv", ".tsv"}:
                loaded = datasets.load_dataset(
                    "csv",
                    data_files=str(path),
                    delimiter="," if ext == ".csv" else "\t",
                )
            else:
                raise ValueError(f"Unsupported file type: {path}")
            return DatasetDict(dict(loaded.items()))
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


def _load_dataset_with_column_union(dataset_name: str) -> DatasetDict:
    """Load a flat CSV dataset while unioning columns across data files.

    This is a fallback path used when the standard :func:`load_dataset` call
    fails due to heterogeneous column sets across CSV shards. The implementation
    fetches every declared data file, reads it via :mod:`pandas`, aligns missing
    columns, and rehydrates the result into a :class:`~datasets.DatasetDict`.

    :param dataset_name: Hugging Face dataset identifier (optionally ``name``).
    :returns: ``DatasetDict`` with the unioned schema.
    :raises RuntimeError: If :mod:`pandas` is unavailable or no data files were
        discovered for the dataset.
    """

    if pd is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            "pandas is required to load '%s' due to column mismatches. Install pandas"
            " to enable the union fallback." % dataset_name
        )

    builder = datasets.load_dataset_builder(dataset_name)
    data_files = getattr(builder.config, "data_files", None)
    if not data_files:
        raise RuntimeError(f"Dataset '{dataset_name}' does not expose data_files metadata")

    # Normalise split -> list[str]
    split_files: Dict[str, list[str]] = {}
    for split_name, files in data_files.items():
        if isinstance(files, str):
            split_files[split_name] = [files]
        else:
            split_files[split_name] = list(files)

    fs = builder._fs  # type: ignore[attr-defined]
    unioned_splits: Dict[str, Dataset] = {}

    for split_name, file_list in split_files.items():
        frames = []
        for file_ref in file_list:
            handle = None
            try:
                open_fs = fs  # type: ignore[attr-defined]
                open_path = file_ref
                if isinstance(file_ref, str) and "://" in file_ref:
                    if url_to_fs is None:
                        raise RuntimeError(
                            "Encountered remote data file '%s' but fsspec is unavailable. "
                            "Install fsspec to enable remote downloads." % file_ref
                        )
                    remote_fs, remote_path = url_to_fs(file_ref)
                    open_fs = remote_fs
                    open_path = remote_path
                try:
                    handle = open_fs.open(open_path, "rb")  # type: ignore[attr-defined]
                except FileNotFoundError:
                    if (
                        url_to_fs is not None
                        and isinstance(file_ref, str)
                        and "://" not in file_ref
                    ):
                        remote_fs, remote_path = url_to_fs(file_ref)
                        handle = remote_fs.open(remote_path, "rb")
                    else:
                        raise
                frame = pd.read_csv(handle)
            finally:
                if handle is not None:
                    handle.close()
            if "Unnamed: 0" in frame.columns:
                frame = frame.drop(columns=["Unnamed: 0"])
            frames.append(frame)

        if not frames:
            continue

        all_columns = sorted({col for frame in frames for col in frame.columns})
        aligned_frames = []
        for frame in frames:
            missing_cols = [col for col in all_columns if col not in frame.columns]
            if missing_cols:
                for col in missing_cols:
                    frame[col] = pd.NA
            aligned_frames.append(frame[all_columns])

        combined = pd.concat(aligned_frames, ignore_index=True)
        unioned_splits[split_name] = Dataset.from_pandas(combined, preserve_index=False)

    if not unioned_splits:
        raise RuntimeError(f"No data could be loaded for dataset '{dataset_name}'")

    return DatasetDict(unioned_splits)


def map_rows_to_examples(
    dataset: DatasetDict,
    *,
    system_prompt: Optional[str],
    sol_key: Optional[str],
    max_history: int,
) -> DatasetDict:
    """Convert interaction rows into cleaned prompt examples.

    :param dataset: Source datasets keyed by split.
    :param system_prompt: Optional system prompt override applied to every row.
    :param sol_key: Alternate column containing the gold next-video identifier.
    :param max_history: Maximum number of prior interactions to embed in ``state_text``.
    :returns: Dataset mapping where each split has been converted to prompt-ready rows.
    """

    mapped = DatasetDict()
    for split_name, split_ds in dataset.items():
        mapped_split = split_ds.map(
            lambda ex, prompt=system_prompt: row_to_example(
                ex,
                prompt,
                sol_key,
                max_history,
            ),
            remove_columns=split_ds.column_names,
            load_from_cache_file=False,
        )
        mapped[split_name] = mapped_split
    return mapped


def ensure_shared_schema(datasets_map: Dict[str, datasets.Dataset]) -> Dict[str, datasets.Dataset]:
    """Align feature schemas across splits so they expose identical columns.

    :param datasets_map: Mapping of split name to dataset requiring schema alignment.
    :returns: New mapping with each split cast to the union of all features.
    """

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

    raw = load_raw(dataset_name, validation_ratio=opts.validation_ratio)

    filtered = filter_prompt_ready(raw, sol_key=opts.sol_key)
    issue_counts = compute_issue_counts(filtered)
    if issue_counts:
        log.info("Issue distribution per split: %s", issue_counts)

    mapped = map_rows_to_examples(
        filtered,
        system_prompt=opts.system_prompt,
        sol_key=opts.sol_key,
        max_history=opts.max_history,
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
    run prompt diagnostics (e.g., prior-history coverage).  Downstream
    consumers that require the historical one-row-per-participant-and-issue
    view can apply this helper to recover the previous behaviour.
    """

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

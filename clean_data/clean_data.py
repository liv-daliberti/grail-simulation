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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

import bz2
import csv
import gzip
import importlib
import io
import logging
import lzma
import zipfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from clean_data.filters import compute_issue_counts, filter_prompt_ready
from clean_data.prompt.constants import REQUIRED_PROMPT_COLUMNS
from clean_data.prompting import row_to_example

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

    class DatasetGenerationCastError(Exception):  # type: ignore
        """Fallback stub for :class:`datasets.builder.DatasetGenerationCastError`.

        :meta private:
        """


def _ensure_datasets_imported() -> None:
    """Ensure the optional datasets dependency (or its stub) is available.

    :raises ModuleNotFoundError: If the optional ``datasets`` package cannot be imported.
    """

    if datasets is not None and Features is not Any:
        return

    module = importlib.import_module("datasets")
    module_globals = globals()
    module_globals["datasets"] = module
    module_globals["Dataset"] = getattr(module, "Dataset", Dataset)
    module_globals["DatasetDict"] = getattr(module, "DatasetDict", DatasetDict)
    module_globals["Features"] = getattr(module, "Features", Features)
    module_globals["HFSequence"] = getattr(module, "Sequence", HFSequence)
    module_globals["Value"] = getattr(module, "Value", Value)
    try:
        builder_module = importlib.import_module("datasets.builder")
    except ModuleNotFoundError:  # pragma: no cover - optional dependency
        return
    module_globals["DatasetGenerationCastError"] = getattr(
        builder_module,
        "DatasetGenerationCastError",
        DatasetGenerationCastError,
    )

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency for linting
    pd = None  # type: ignore

try:
    from fsspec.core import url_to_fs
except ImportError:  # pragma: no cover - optional dependency for linting
    url_to_fs = None  # type: ignore

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


@dataclass(frozen=True)
class _ReadAttempt:
    """Capture a single pandas CSV parsing attempt configuration.

    :param engine: Pandas CSV engine identifier (``"c"`` or ``"python"``).
    :param sep: Delimiter candidate evaluated for the CSV payload.
    :param on_bad_lines: Policy forwarded to pandas for malformed rows.
    """

    engine: str
    sep: Optional[str]
    on_bad_lines: Optional[str] = None


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

    _ensure_datasets_imported()

    if pd is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            f"pandas is required to load '{dataset_name}' due to column mismatches. "
            "Install pandas to enable the union fallback."
        )

    builder = datasets.load_dataset_builder(dataset_name)
    data_files = getattr(builder.config, "data_files", None)
    if not data_files:
        raise RuntimeError(f"Dataset '{dataset_name}' does not expose data_files metadata")

    split_files = _normalise_split_mappings(data_files)
    features: Optional[Features] = getattr(builder.info, "features", None)
    loader = _ColumnUnionLoader(dataset_name, builder, features)
    unioned = loader.build(split_files)
    if not unioned:
        raise RuntimeError(f"No data could be loaded for dataset '{dataset_name}'")
    return DatasetDict(unioned)


def _normalise_split_mappings(data_files: Dict[str, Any]) -> Dict[str, list[str]]:
    """Coerce the builder ``data_files`` mapping into ``split -> [files]`` form.

    :param data_files: Mapping extracted from ``datasets.Builder`` configuration.
    :returns: Normalised mapping where every value is a list of file references.
    :raises RuntimeError: If a mapping value cannot be interpreted as str or sequence.
    """

    split_files: Dict[str, list[str]] = {}
    for split_name, files in data_files.items():
        if isinstance(files, str):
            split_files[split_name] = [files]
        elif isinstance(files, Sequence):
            split_files[split_name] = list(files)
        else:
            raise RuntimeError(
                f"Unsupported data_files entry for split '{split_name}': {type(files)!r}"
            )
    return split_files


@dataclass
class _ColumnUnionLoader:
    """Helper that recreates CSV datasets while unioning distinct column sets.

    :param dataset_name: Hugging Face dataset identifier being reconstructed.
    :param builder: Dataset builder instance supplying data files and filesystem.
    :param features: Optional feature schema used to cast the reconstructed frames.
    :ivar expected_columns: Columns inferred from the declared feature schema.
    :ivar filesystem: Filesystem handle used to read the dataset shards.
    """

    dataset_name: str
    builder: Any
    features: Optional[Features]
    expected_columns: set[str] = field(init=False)
    filesystem: Any = field(init=False)

    def __post_init__(self) -> None:
        """Populate derived filesystem metadata for dataset reconstruction.

        :raises RuntimeError: If the dataset builder does not expose a filesystem handle.
        """
        if isinstance(self.features, Features):
            self.expected_columns = set(self.features.keys())
        else:
            self.expected_columns = set()
        filesystem_handle = getattr(self.builder, "_fs", None)
        if filesystem_handle is None:  # pragma: no cover - defensive branch
            raise RuntimeError(
                f"Dataset builder for '{self.dataset_name}' does not expose a filesystem handle"
            )
        self.filesystem = filesystem_handle

    def build(self, split_files: Dict[str, list[str]]) -> Dict[str, Dataset]:
        """Union column schemas for every split and return the reconstructed datasets.

        :param split_files: Mapping of split name to the list of backing data files.
        :returns: Dictionary of split names to rehydrated ``Dataset`` objects.
        """
        unioned: Dict[str, Dataset] = {}
        for split_name, file_refs in split_files.items():
            dataset = self._build_split(split_name, file_refs)
            if dataset is not None:
                unioned[split_name] = dataset
        return unioned

    def _build_split(self, split_name: str, file_refs: list[str]) -> Optional[Dataset]:
        """Reconstruct a single split by unioning column schemas across files.

        :param split_name: Identifier of the split being reconstructed.
        :param file_refs: Iterable of file references associated with the split.
        :returns: Hugging Face ``Dataset`` instance when any rows are loaded, otherwise ``None``.
        """
        frames: list[pd.DataFrame] = []
        for file_ref in file_refs:
            filesystem, path = self._resolve_file_ref(file_ref)
            frame = self._read_csv_frame(filesystem, path)
            frame = self._postprocess_frame(frame)
            frames.append(frame)

        if not frames:
            log.debug("No frames produced for split '%s'", split_name)
            return None

        combined = self._combine_frames(frames)
        combined = self._apply_feature_casts(combined)
        return Dataset.from_pandas(combined, preserve_index=False)

    def _resolve_file_ref(self, file_ref: Any) -> Tuple[Any, Any]:
        """Resolve a file reference to an ``fsspec`` filesystem and path.

        :param file_ref: Local or remote identifier describing a dataset shard.
        :returns: Tuple of filesystem handle and path within that filesystem.
        :raises RuntimeError: If remote references are encountered without ``fsspec``.
        :raises FileNotFoundError: If the resolved file cannot be opened.
        """
        if isinstance(file_ref, str) and "://" in file_ref:
            if url_to_fs is None:
                raise RuntimeError(
                    f"Encountered remote data file '{file_ref}' but fsspec is unavailable. "
                    "Install fsspec to enable remote downloads."
                )
            return url_to_fs(file_ref)

        try:
            with self.filesystem.open(file_ref, "rb"):
                return self.filesystem, file_ref
        except FileNotFoundError:
            if url_to_fs is not None and isinstance(file_ref, str):
                return url_to_fs(file_ref)
            raise

    def _read_csv_frame(self, filesystem: Any, path: Any) -> pd.DataFrame:
        """Read and normalise a CSV shard into a pandas DataFrame.

        :param filesystem: Filesystem handle capable of opening the shard.
        :param path: Path or identifier of the shard within ``filesystem``.
        :returns: DataFrame containing the parsed rows.
        :raises RuntimeError: If none of the decoding strategies succeed.
        """
        with filesystem.open(path, "rb") as raw_handle:
            raw_bytes = raw_handle.read()

        data_bytes = self._decompress_payload(raw_bytes, path)
        sample_bytes = data_bytes[:16384]
        decoded_samples = self._decode_sample_texts(sample_bytes)

        for encoding, sample_text in decoded_samples:
            for attempt in self._build_attempts(sample_text):
                try:
                    return self._read_with_attempt(data_bytes, encoding, attempt)
                except UnicodeDecodeError:
                    break
                except (pd.errors.ParserError, pd.errors.EmptyDataError, ValueError, OSError):
                    continue

        raise RuntimeError(f"Unable to read CSV file '{path}' using available fallbacks")

    @staticmethod
    def _decompress_payload(payload: bytes, path: Any) -> bytes:
        """Expand compressed payloads emitted by common archival formats.

        :param payload: Raw bytes read from the filesystem.
        :param path: Identifier of the current shard, used for diagnostics.
        :returns: Decompressed bytes when compression is detected, otherwise the original payload.
        """
        try:
            if payload.startswith(b"\x1f\x8b\x08"):
                return gzip.decompress(payload)
            if payload.startswith(b"PK\x03\x04"):
                with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                    for name in archive.namelist():
                        if not name.endswith("/"):
                            return archive.read(name)
                log.debug("Zip archive '%s' does not contain file entries; using raw bytes", path)
                return payload
            if payload.startswith(b"BZh"):
                return bz2.decompress(payload)
            if payload.startswith(b"\xfd7zXZ\x00") or payload.startswith(b"\x5d\x00\x00"):
                return lzma.decompress(payload)
        except (OSError, zipfile.BadZipFile, lzma.LZMAError) as err:
            log.debug(
                (
                    "Failed to decompress '%s' despite matching signature; falling back to raw "
                    "bytes (%s)"
                ),
                path,
                err,
            )
        return payload

    @staticmethod
    def _decode_sample_texts(sample_bytes: bytes) -> list[tuple[str, str]]:
        """Decode sample bytes using several fallback encodings.

        :param sample_bytes: Initial slice of the CSV payload used for detection heuristics.
        :returns: List of ``(encoding, text)`` tuples in evaluation order.
        :raises UnicodeDecodeError: If no encoding successfully decodes the bytes.
        :raises LookupError: If no recognised encoding can be used during detection.
        """
        decoded: list[tuple[str, str]] = []
        last_decode_err: Optional[UnicodeDecodeError] = None
        last_lookup_err: Optional[LookupError] = None

        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                decoded.append((encoding, sample_bytes.decode(encoding)))
            except UnicodeDecodeError as err:
                last_decode_err = err
            except LookupError as err:
                last_lookup_err = err

        if decoded:
            return decoded
        if last_decode_err is not None:
            raise last_decode_err
        if last_lookup_err is not None:
            raise last_lookup_err
        raise UnicodeDecodeError("unicodeescape", b"", 0, 1, "no valid encoding candidates found")

    def _build_attempts(self, sample_text: str) -> list[_ReadAttempt]:
        """Construct CSV parsing attempts from the sample payload.

        :param sample_text: Text decoded from the beginning of the CSV payload.
        :returns: Ordered list of parsing attempts to evaluate.
        """
        delimiters = self._candidate_delimiters(sample_text)
        attempts: list[_ReadAttempt] = []
        seen: set[tuple[str, Optional[str], Optional[str]]] = set()

        for delimiter in delimiters:
            for engine in ("c", "python"):
                key = (engine, delimiter, None)
                if key not in seen:
                    attempts.append(_ReadAttempt(engine=engine, sep=delimiter))
                    seen.add(key)

        for on_bad_lines in (None, "skip"):
            key = ("python", None, on_bad_lines)
            if key not in seen:
                attempts.append(_ReadAttempt(engine="python", sep=None, on_bad_lines=on_bad_lines))
                seen.add(key)

        return attempts

    @staticmethod
    def _candidate_delimiters(sample_text: str) -> list[str]:
        """Infer potential delimiter characters from sample CSV text.

        :param sample_text: Text snippet used to identify likely delimiters.
        :returns: List of delimiter characters ranked by confidence.
        """
        try:
            sniffed = csv.Sniffer().sniff(
                sample_text,
                delimiters=[",", "\t", ";", "|", "\x1f"],
            )
            primary = [sniffed.delimiter]
        except csv.Error:
            primary = []
        fallbacks = [",", "\t", ";", "|", "\x1f"]
        return primary + fallbacks

    @staticmethod
    def _read_with_attempt(data_bytes: bytes, encoding: str, attempt: _ReadAttempt) -> pd.DataFrame:
        """Evaluate a single pandas CSV read attempt.

        :param data_bytes: Raw CSV payload (optionally decompressed).
        :param encoding: Text encoding used for decoding the payload.
        :param attempt: Candidate configuration describing the CSV read.
        :returns: DataFrame produced by ``pandas.read_csv`` when successful.
        """
        kwargs: Dict[str, Any] = {
            "encoding": encoding,
            "low_memory": False,
            "engine": attempt.engine,
        }
        if attempt.sep is not None:
            kwargs["sep"] = attempt.sep
        if attempt.on_bad_lines and attempt.engine == "python":
            kwargs["on_bad_lines"] = attempt.on_bad_lines
        return pd.read_csv(io.BytesIO(data_bytes), **kwargs)

    def _postprocess_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Rename and prune columns so they align with the expected schema.

        :param frame: DataFrame produced by pandas for an individual shard.
        :returns: Normalised DataFrame ready for unioning.
        """
        frame = frame.drop(columns=["Unnamed: 0"], errors="ignore")
        frame_columns = set(frame.columns)
        rename_map = {}
        for column in frame.columns:
            new_name = self._maybe_canonical_name(column, frame_columns)
            if new_name != column:
                rename_map[column] = new_name
        if rename_map:
            frame = frame.rename(columns=rename_map)
        return frame

    def _maybe_canonical_name(self, column: str, frame_columns: set[str]) -> str:
        """Return a canonical column name when possible.

        :param column: Column name emitted by the CSV file.
        :param frame_columns: All columns currently present in the frame.
        :returns: Either the canonical column name or the original value when no change is needed.
        """
        if "_pre" not in column:
            return column

        candidates = self._candidate_column_names(column)
        for candidate in candidates:
            if not candidate or candidate == column:
                continue
            if candidate in frame_columns:
                continue
            if self.expected_columns and candidate in self.expected_columns:
                return candidate
            if not self.expected_columns:
                return candidate
        return column

    @staticmethod
    def _candidate_column_names(column: str) -> list[str]:
        """Generate canonical column name candidates for ``_pre`` variants.

        :param column: Column name sourced from the CSV shard.
        :returns: Candidate column names stripped of ``_pre`` suffixes.
        """
        candidates: list[str] = []
        if column.endswith("_pre"):
            candidates.append(column[:-4])
        if "_pre_" in column:
            candidates.append(column.replace("_pre_", "_", 1))
        if "_pre" in column:
            candidates.append(column.replace("_pre", "", 1))
        return [candidate.strip("_") for candidate in candidates]

    @staticmethod
    def _combine_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
        """Align columns across frames and concatenate them.

        :param frames: Collection of DataFrames produced from dataset shards.
        :returns: DataFrame containing the union of the provided columns.
        """
        all_columns = sorted({col for frame in frames for col in frame.columns})
        aligned = [frame.reindex(columns=all_columns, fill_value=pd.NA) for frame in frames]
        return pd.concat(aligned, ignore_index=True)

    def _apply_feature_casts(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Cast string-like columns so they align with the dataset feature schema.

        :param frame: DataFrame produced by combining all shard outputs.
        :returns: DataFrame with corrected pandas dtypes.
        """
        if frame.empty:
            return frame

        object_columns = frame.select_dtypes(include="object").columns
        for column_name in object_columns:
            feature = None
            if isinstance(self.features, Features):
                feature = self.features.get(column_name)
            expected_dtype = getattr(feature, "dtype", None)
            if feature is None or expected_dtype == "string":
                frame[column_name] = frame[column_name].astype("string")

        if isinstance(self.features, Features):
            for column_name, feature in self.features.items():
                if not isinstance(feature, Value):
                    continue
                if column_name not in frame.columns:
                    continue
                if feature.dtype == "string":
                    frame[column_name] = frame[column_name].astype("string")
        return frame


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
        mapped_split = split_ds.map(
            lambda ex, prompt=system_prompt: row_to_example(
                ex,
                prompt,
                sol_key,
                max_history,
            ),
            remove_columns=split_ds.column_names,
            load_from_cache_file=False,
            num_proc=num_proc,
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

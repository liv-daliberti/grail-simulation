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

"""Dataset loading utilities shared by ``clean_data.clean_data``.

This module centralizes optional ``datasets`` imports, fallback CSV reconstruction,
and schema-alignment utilities used when building cleaned prompt datasets.
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
from typing import Any, Dict, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency
    import datasets  # type: ignore
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

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional dependency for linting
    pd = None  # type: ignore

try:
    from fsspec.core import url_to_fs
except ImportError:  # pragma: no cover - optional dependency for linting
    url_to_fs = None  # type: ignore

log = logging.getLogger("clean_grail")


def ensure_datasets_imported() -> None:
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


def normalize_split_mappings(data_files: Dict[str, Any]) -> Dict[str, list[str]]:
    """Coerce the builder ``data_files`` mapping into ``split -> [files]`` form.

    :param data_files: Mapping extracted from ``datasets.Builder`` configuration.
    :returns: Normalized mapping where every value is a list of file references.
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
    """

    dataset_name: str
    builder: Any
    features: Optional[Features]
    expected_columns: set[str] = field(init=False)
    filesystem: Any = field(init=False)

    def __post_init__(self) -> None:
        """Populate derived filesystem metadata for dataset reconstruction."""

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
        return datasets.Dataset.from_pandas(combined, preserve_index=False)

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
        """Read and normalize a CSV shard into a pandas DataFrame.

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
            log.debug("Failed to decompress '%s'; using raw bytes (%s)", path, err)
        return payload

    @staticmethod
    def _decode_sample_texts(sample_bytes: bytes) -> list[tuple[str, str]]:
        """Decode sample bytes using several fallback encodings.

        :param sample_bytes: Initial slice of the CSV payload used for detection heuristics.
        :returns: List of ``(encoding, text)`` tuples in evaluation order.
        :raises UnicodeDecodeError: If no encoding successfully decodes the bytes.
        :raises LookupError: If no recognized encoding can be used during detection.
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
        :returns: Normalized DataFrame ready for unioning.
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
            if candidate in frame_columns or candidate in self.expected_columns:
                return candidate
        return column

    @staticmethod
    def _candidate_column_names(column: str) -> list[str]:
        """Generate canonical column name candidates for ``_pre`` variants.

        :param column: Column name sourced from the CSV shard.
        :returns: Candidate column names stripped of ``_pre`` suffixes.
        """

        if not column.endswith("_pre"):
            return []
        base = column[: -len("_pre")]
        return [base, base.rstrip("_"), base + "_orig"]

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


def load_dataset_with_column_union(dataset_name: str) -> DatasetDict:
    """Load a flat CSV dataset while unioning columns across data files.

    :param dataset_name: Hugging Face dataset identifier to load.
    :returns: ``DatasetDict`` exposing the unioned schema.
    :raises RuntimeError: If pandas is unavailable or the dataset lacks ``data_files`` metadata.
    """

    ensure_datasets_imported()

    if pd is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            f"pandas is required to load '{dataset_name}' due to column mismatches. "
            "Install pandas to enable the union fallback."
        )

    builder = datasets.load_dataset_builder(dataset_name)
    data_files = getattr(builder.config, "data_files", None)
    if not data_files:
        raise RuntimeError(f"Dataset '{dataset_name}' does not expose data_files metadata")

    split_files = normalize_split_mappings(data_files)
    features: Optional[Features] = getattr(builder.info, "features", None)
    loader = _ColumnUnionLoader(dataset_name, builder, features)
    unioned = loader.build(split_files)
    if not unioned:
        raise RuntimeError(f"No data could be loaded for dataset '{dataset_name}'")
    return DatasetDict(unioned)


__all__ = [
    "Dataset",
    "DatasetDict",
    "DatasetGenerationCastError",
    "Features",
    "HFSequence",
    "Value",
    "datasets",
    "ensure_datasets_imported",
    "load_dataset_with_column_union",
    "normalize_split_mappings",
    "pd",
    "url_to_fs",
]

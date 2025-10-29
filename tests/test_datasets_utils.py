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

"""Unit tests for :mod:`clean_data._datasets` utilities."""

from __future__ import annotations

import bz2
import gzip
import io
import lzma
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from clean_data import _datasets as datasets_mod


class _DummyFS:
    """Filesystem shim that delegates to Python's built-in file handling."""

    def open(self, path: str, mode: str = "rb") -> Any:
        return open(path, mode)  # noqa: P201  (context managed by caller)


def _make_builder(tmp_path: Path, files: dict[str, list[list[tuple[str, Any]]]]) -> Any:
    """Create a fake datasets builder with real CSV payloads on disk."""

    data_files: dict[str, list[str]] = {}
    for split, rows in files.items():
        paths: list[str] = []
        for idx, (columns, values) in enumerate(rows):
            frame = pd.DataFrame(values, columns=columns)
            file_path = tmp_path / f"{split}_{idx}.csv"
            frame.to_csv(file_path, index=False)
            paths.append(str(file_path))
        data_files[split] = paths

    builder = SimpleNamespace(
        _fs=_DummyFS(),
        config=SimpleNamespace(data_files=data_files),
        info=SimpleNamespace(features=None),
    )
    return builder


def test_column_union_loader_unions_and_casts(tmp_path, monkeypatch):
    """The column union loader should merge disjoint columns per split."""

    builder = _make_builder(
        tmp_path,
        {
            "train": [
                (["col_a", "col_shared"], [[1, "a"], [2, "b"]]),
                (["col_shared", "col_c"], [["x", 3.14], ["y", 2.71]]),
            ]
        },
    )
    captured_frames: list[pd.DataFrame] = []

    def fake_from_pandas(frame: pd.DataFrame, *, preserve_index: bool) -> str:
        captured_frames.append(frame)
        return "mock_dataset"

    monkeypatch.setattr(datasets_mod.datasets.Dataset, "from_pandas", staticmethod(fake_from_pandas))

    loader = datasets_mod._ColumnUnionLoader("dummy", builder, features=None)
    unioned = loader.build(builder.config.data_files)

    assert unioned == {"train": "mock_dataset"}
    assert len(captured_frames) == 1
    frame = captured_frames[0]
    assert set(frame.columns) == {"col_a", "col_shared", "col_c"}
    # Column that was missing in one shard should be NA filled.
    assert frame["col_a"].isna().sum() == 2
    assert frame["col_c"].isna().sum() == 2


def test_column_union_loader_returns_none_for_empty_split(monkeypatch):
    """When a split has no files, the loader should return ``None``."""

    builder = SimpleNamespace(_fs=_DummyFS())
    loader = datasets_mod._ColumnUnionLoader("dummy", builder, features=None)
    dataset = loader._build_split("train", [])
    assert dataset is None


def test_column_union_loader_remote_requires_fsspec(monkeypatch):
    """Remote references should raise when ``fsspec`` is unavailable."""

    builder = SimpleNamespace(_fs=_DummyFS())
    loader = datasets_mod._ColumnUnionLoader("dummy", builder, features=None)
    monkeypatch.setattr(datasets_mod, "url_to_fs", None)
    with pytest.raises(RuntimeError):
        loader._resolve_file_ref("https://example.com/data.csv")


def test_decompress_payload_variants(tmp_path, monkeypatch):
    """Compression helpers should expand gzip/zip/bz2/xz payloads."""

    original = b"hello world"

    gzip_bytes = io.BytesIO()
    with gzip.GzipFile(fileobj=gzip_bytes, mode="wb") as handle:
        handle.write(original)
    assert datasets_mod._ColumnUnionLoader._decompress_payload(gzip_bytes.getvalue(), "gz") == original

    zip_bytes = io.BytesIO()
    with zipfile.ZipFile(zip_bytes, mode="w") as archive:
        archive.writestr("payload.txt", original)
    assert datasets_mod._ColumnUnionLoader._decompress_payload(zip_bytes.getvalue(), "zip") == original

    assert datasets_mod._ColumnUnionLoader._decompress_payload(bz2.compress(original), "bz2") == original
    assert datasets_mod._ColumnUnionLoader._decompress_payload(lzma.compress(original), "xz") == original

    # Malformed archives should fall back to the raw bytes.
    assert (
        datasets_mod._ColumnUnionLoader._decompress_payload(b"PK\x03\x04bad", "zip") == b"PK\x03\x04bad"
    )


def test_decode_sample_texts_and_build_attempts(tmp_path):
    """Sample decoders should evaluate multiple encodings and delimiters."""

    samples = datasets_mod._ColumnUnionLoader._decode_sample_texts(b"col1,col2\n1,2\n")
    encodings = [encoding for encoding, _ in samples]
    assert encodings[0] == "utf-8"

    builder = _make_builder(tmp_path, {"train": [(["col"], [[1]])]})
    loader = datasets_mod._ColumnUnionLoader("dummy", builder, features=None)
    attempts = loader._build_attempts(samples[0][1])
    # Expect multiple delimiter/engine combinations, starting with comma/"c".
    assert attempts[0].engine == "c"
    assert attempts[0].sep == ","
    assert any(attempt.on_bad_lines == "skip" for attempt in attempts if attempt.engine == "python")


def test_load_dataset_with_column_union_requires_pandas(monkeypatch):
    """When pandas is absent, the union loader should raise immediately."""

    monkeypatch.setattr(datasets_mod, "pd", None)
    with pytest.raises(RuntimeError):
        datasets_mod.load_dataset_with_column_union("dummy/dataset")


def test_load_dataset_with_column_union_invokes_loader(tmp_path, monkeypatch):
    """The public loader should normalize data_files and return unioned splits."""

    builder = _make_builder(
        tmp_path,
        {
            "train": [
                (["col_a"], [[1], [2]]),
            ],
            "validation": [
                (["col_b"], [[3], [4]]),
            ],
        },
    )
    builder.info.features = None

    monkeypatch.setattr(datasets_mod, "pd", pd)
    monkeypatch.setattr(datasets_mod, "datasets", datasets_mod.datasets)
    monkeypatch.setattr(datasets_mod.datasets, "load_dataset_builder", lambda _: builder)

    captured_splits = {"train": "union-train", "validation": "union-validation"}

    class RecordingLoader(datasets_mod._ColumnUnionLoader):  # type: ignore[misc]
        def build(self, split_files):
            assert set(split_files.keys()) == {"train", "validation"}
            return captured_splits

    monkeypatch.setattr(datasets_mod, "_ColumnUnionLoader", RecordingLoader)

    result = datasets_mod.load_dataset_with_column_union("dummy/dataset")
    assert result == captured_splits


def test_load_dataset_with_column_union_missing_data_files(monkeypatch):
    """Missing ``data_files`` metadata should surface a runtime error."""

    builder = SimpleNamespace(
        _fs=_DummyFS(),
        config=SimpleNamespace(data_files=None),
        info=SimpleNamespace(features=None),
    )
    monkeypatch.setattr(datasets_mod, "pd", pd)
    monkeypatch.setattr(datasets_mod, "datasets", datasets_mod.datasets)
    monkeypatch.setattr(datasets_mod.datasets, "load_dataset_builder", lambda _: builder)

    with pytest.raises(RuntimeError):
        datasets_mod.load_dataset_with_column_union("dummy/dataset")

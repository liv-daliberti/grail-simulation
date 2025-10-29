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

"""Tests for high-level helpers exposed by :mod:`clean_data.clean_data`."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import clean_data.clean_data as clean_module
from clean_data._datasets import DatasetDict


class _MockDataset:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self.column_names = sorted(rows[0].keys()) if rows else []
        self.features = {}

    def map(self, fn, **kwargs):
        return _MockDataset([fn(row) for row in self._rows])

    def filter(self, fn, **kwargs):
        return _MockDataset([row for row in self._rows if fn(row)])

    def add_column(self, name, values):
        for row, value in zip(self._rows, values):
            row[name] = value
        return self

    def cast(self, features):
        self.features = features
        return self

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, item):
        return self._rows[item]

    def items(self):
        return self._rows

    def to_pandas(self):
        return pd.DataFrame(self._rows)


@pytest.fixture
def dataset_dict():
    return DatasetDict({"train": _MockDataset([{"x": 1, "y": 0}, {"x": 2, "y": 1}])})


def test_load_raw_prefers_codeocean_when_module_present(tmp_path, monkeypatch):
    capsule_path = tmp_path / "capsule"
    capsule_path.mkdir()

    def fake_import(name):
        if name == "clean_data.codeocean":
            return SimpleNamespace(load_codeocean_dataset=lambda path, validation_ratio: "capsule-dataset")
        raise ModuleNotFoundError

    monkeypatch.setattr(clean_module, "_try_load_codeocean_dataset", lambda path, ratio: "capsule-dataset")
    result = clean_module.load_raw(str(capsule_path), validation_ratio=0.2)
    assert result == "capsule-dataset"


def test_load_raw_falls_back_to_file_type(tmp_path, monkeypatch):
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps({"train": [{"col": 1}]}))

    monkeypatch.setattr(clean_module, "datasets", SimpleNamespace(load_dataset=lambda *args, **kwargs: {"train": []}))
    monkeypatch.setattr(clean_module, "DatasetDict", DatasetDict)

    result = clean_module.load_raw(str(json_file))
    assert isinstance(result, DatasetDict)
    with pytest.raises(ValueError):
        clean_module.load_raw(str(tmp_path / "data.unsupported"))


def test_load_raw_union_loader(monkeypatch):
    def failing_loader(*args, **kwargs):
        raise clean_module.DatasetGenerationCastError("boom")

    monkeypatch.setattr(clean_module.datasets, "load_dataset", failing_loader)
    monkeypatch.setattr(clean_module, "_load_dataset_with_column_union", lambda name: "union-result")

    assert clean_module.load_raw("dummy") == "union-result"


def test_map_rows_to_examples_validates_num_proc(dataset_dict):
    with pytest.raises(ValueError):
        clean_module.map_rows_to_examples(dataset_dict, system_prompt=None, sol_key=None, max_history=1, num_proc=0)


@pytest.mark.parametrize(
    "sol_key,system_prompt",
    [
        ("alt_gold", None),
        (None, "SYSTEM"),
    ],
)
def test_map_rows_to_examples_sol_key_and_system(monkeypatch, sol_key, system_prompt):
    dataset = DatasetDict({"train": _MockDataset([{"slate_items_json": [], "alt_gold": "123", "current_video_id": ""}])})

    def stub_row_to_example(row, prompt, sol, hist):
        return {"prompt": prompt, "gold": sol}

    monkeypatch.setattr(clean_module, "row_to_example", stub_row_to_example)
    mapped = clean_module.map_rows_to_examples(dataset, system_prompt=system_prompt, sol_key=sol_key, max_history=1)
    result = mapped["train"].items()[0]
    expected_sol = sol_key if sol_key else None
    assert result["prompt"] == system_prompt
    assert result["gold"] == expected_sol


def test_ensure_shared_schema_handles_sequences(monkeypatch):
    seq_feature = clean_module.HFSequence(clean_module.Value("string"))
    dataset_a = SimpleNamespace(
        column_names=["a"],
        features={"a": seq_feature},
        add_column=lambda *args, **kwargs: dataset_a,
        cast=lambda features: SimpleNamespace(features=features),
    )
    dataset_b = SimpleNamespace(
        column_names=[],
        features={},
        add_column=lambda name, filler: dataset_b,
        cast=lambda features: SimpleNamespace(features=features),
    )
    merged = clean_module.ensure_shared_schema({"a": dataset_a, "b": dataset_b})
    assert set(merged.keys()) == {"a", "b"}


def test_build_clean_dataset_dedupe_and_options(monkeypatch):
    raw_dataset = DatasetDict(
        {
            "train": _MockDataset(
                [
                    {"participant_id": "p1", "issue": "gun_control", "slate_items_json": [], "current_video_id": ""},
                    {"participant_id": "p1", "issue": "gun_control", "slate_items_json": [], "current_video_id": ""},
                ]
            )
        }
    )

    monkeypatch.setattr(clean_module, "load_raw", lambda name, validation_ratio: raw_dataset)
    monkeypatch.setattr(clean_module, "map_rows_to_examples", lambda ds, **opts: ds)
    monkeypatch.setattr(clean_module, "filter_prompt_ready", lambda ds, **opts: ds)
    monkeypatch.setattr(clean_module, "compute_issue_counts", lambda ds: {})

    def fake_generate_stats(ds, output_dir, **kwargs):
        raise ImportError("stats missing")

    monkeypatch.setattr(clean_module, "generate_prompt_stats", fake_generate_stats)

    opts = clean_module.BuildOptions(system_prompt="SYS", sol_key="alt", max_history=2)
    result = clean_module.build_clean_dataset("dummy", options=opts)
    assert isinstance(result, DatasetDict)


def test_export_issue_datasets(tmp_path, monkeypatch):
    dataset = DatasetDict({"train": _MockDataset([{"issue": "gun_control", "value": 1}]), "validation": _MockDataset([])})
    out_dir = tmp_path / "out"

    saved = []

    class FakeDataset(_MockDataset):
        def save_to_disk(self, path):
            saved.append(Path(path))

    dataset["train"] = FakeDataset(dataset["train"]._rows)

    monkeypatch.setattr(clean_module, "DatasetDict", DatasetDict)

    clean_module.export_issue_datasets(dataset, out_dir, {"gun_control": "repo"}, push_to_hub=False)
    assert saved


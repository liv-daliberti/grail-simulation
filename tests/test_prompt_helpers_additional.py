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

"""Additional unit tests for prompt helper utilities and CLI wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import pandas as pd
import pytest

from clean_data.prompt import cli as prompt_cli
from clean_data.prompt import prompting
from clean_data.prompt.summary import summarize_feature
from clean_data.prompt.utils import SeriesPair


def test_viewer_profile_formatters_and_sentence():
    """Ensure demographic helpers produce readable fragments."""

    assert prompting._format_age("30") == "30-year-old"
    assert prompting._format_gender("female") == "woman"
    assert prompting._format_gender("Non-binary") == "Non-Binary"
    assert prompting._format_party_affiliation("Democrat", "Liberal") == "democrat liberal"

    example = {
        "age": "45",
        "q26": "male",
        "q29": "Latino",
        "pid1": "Independent",
        "ideo1": "Moderate",
        "q31": "75000",
        "college": "yes",
        "freq_youtube": "5",
    }
    sentence = prompting._synthesize_viewer_sentence(example)
    assert "45-year-old" in sentence
    assert "man" in sentence
    assert "Latino" in sentence
    assert "watches YouTube" in sentence


def test_resolve_slate_text_falls_back_when_missing():
    """A missing ``slate_text`` should be synthesized from slate items."""

    example = {"slate_text": ""}
    items = [
        {"title": "First option"},
        {"id": "second_option"},
    ]
    text = prompting._resolve_slate_text(example, items)
    assert "1. First option" in text
    assert "2. second_option" in text


def test_summarize_feature_numeric_and_categorical(monkeypatch, tmp_path):
    """summarize_feature should detect numeric vs categorical features."""

    hist_calls: List[str] = []

    monkeypatch.setattr(prompt_cli, "plot_numeric_hist", lambda *args, **kwargs: hist_calls.append("numeric"))
    monkeypatch.setattr(prompt_cli, "plot_categorical_hist", lambda *args, **kwargs: hist_calls.append("categorical"))

    train_df = pd.DataFrame({"value": [1, 2, 3], "issue": ["gun_control"] * 3})
    val_df = pd.DataFrame({"value": [2, 4], "issue": ["gun_control", "gun_control"]})
    pair = SeriesPair(train_df["value"], val_df["value"], train_df, val_df)
    out = summarize_feature(pair, "Value", tmp_path / "value.png")
    assert set(out.keys()) == {"train", "validation"}
    assert hist_calls[-1] == "numeric"

    hist_calls.clear()
    train_df = pd.DataFrame({"cat": ["a", "b", "b"], "issue": ["gun_control"] * 3})
    val_df = pd.DataFrame({"cat": ["a"], "issue": ["gun_control"]})
    pair = SeriesPair(train_df["cat"], val_df["cat"], train_df, val_df)
    out = summarize_feature(pair, "Category", tmp_path / "cat.png")
    assert out["train"]["b"] == 2
    assert hist_calls[-1] == "categorical"


def test_generate_prompt_feature_report_writes_files(monkeypatch, tmp_path):
    """CLI helper should emit summary.json and README.md when splits present."""

    df_train = pd.DataFrame(
        {
            "issue": ["gun_control", "gun_control"],
            "participant_study": ["study1", "study1"],
            "n_options": [2, 3],
            "viewer_profile_sentence": ["sentence", "sentence"],
        }
    )
    df_val = pd.DataFrame(
        {
            "issue": ["gun_control"],
            "participant_study": ["study1"],
            "n_options": [2],
            "viewer_profile_sentence": ["sentence"],
        }
    )

    class Split:
        def __init__(self, df: pd.DataFrame):
            self._df = df

        def to_pandas(self) -> pd.DataFrame:
            return self._df.copy()

    dataset = {"train": Split(df_train), "validation": Split(df_val)}

    monkeypatch.setattr(prompt_cli, "summarize_features", lambda *args, **kwargs: ({"feat": {}}, []))
    monkeypatch.setattr(prompt_cli, "profile_summary", lambda *args, **kwargs: {"train": {"rows": 1, "missing_profile": 0}, "validation": {"rows": 1, "missing_profile": 0}})
    monkeypatch.setattr(prompt_cli, "prior_history_summary", lambda *args, **kwargs: ({"train": {}, "validation": {}}, tmp_path / "prior.png"))
    monkeypatch.setattr(prompt_cli, "n_options_summary", lambda *args, **kwargs: ({"train": {}, "validation": {}}, tmp_path / "nopt.png"))
    monkeypatch.setattr(prompt_cli, "demographic_missing_summary", lambda *args, **kwargs: ({"train": {"total": 1, "missing": 0, "share": 0.0}, "validation": {"total": 1, "missing": 0, "share": 0.0}, "overall": {"total": 2, "missing": 0, "share": 0.0}}, tmp_path / "demo.png"))
    monkeypatch.setattr(prompt_cli, "unique_content_counts", lambda *args, **kwargs: {"train": {}, "validation": {}, "overall": {}})
    monkeypatch.setattr(prompt_cli, "participant_counts_summary", lambda *args, **kwargs: {"train": {}, "validation": {}, "overall": {}})
    monkeypatch.setattr(prompt_cli, "build_markdown_report", lambda ctx: ["# Report", "line"])

    prompt_cli.generate_prompt_feature_report(dataset, tmp_path)
    assert (tmp_path / "summary.json").exists()
    assert json.loads((tmp_path / "summary.json").read_text())["feature_summary"] == {"feat": {}}
    assert (tmp_path / "README.md").exists()


def test_prompt_cli_main_invokes_generate(monkeypatch, tmp_path):
    """The CLI entry point should load the dataset and invoke the generator."""

    called = {}

    monkeypatch.setattr(prompt_cli, "load_dataset_any", lambda path: {"train": [], "validation": []})
    monkeypatch.setattr(
        prompt_cli,
        "generate_prompt_feature_report",
        lambda dataset, output_dir, train_split, validation_split: called.setdefault("invoked", (dataset, output_dir, train_split, validation_split)),
    )

    prompt_cli.main(
        [
            "--dataset",
            "dummy",
            "--output-dir",
            str(tmp_path),
            "--train-split",
            "train",
            "--validation-split",
            "validation",
        ]
    )
    assert "invoked" in called


def test_prompt_stats_main_delegates(monkeypatch):
    """prompt_stats main should delegate to prompt CLI's main entry point."""

    called = {}
    monkeypatch.setattr(prompt_cli, "generate_prompt_feature_report", lambda dataset, output_dir, train_split, validation_split: called.setdefault("report", True))
    dataset = {"train": [], "validation": []}
    monkeypatch.setattr(prompt_cli, "load_dataset_any", lambda _: dataset)

    prompt_cli.generate_prompt_feature_report(dataset, Path.cwd())
    assert called["report"]

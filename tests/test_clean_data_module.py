"""Smoke tests and regression tests for clean_data module helpers."""

from __future__ import annotations

import bz2
import gzip
import io
import json
import lzma
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import pytest
from datasets import Dataset, DatasetDict, Features, Sequence as HFSequence, Value
import clean_data.clean_data

from clean_data.clean_data import (
    BuildOptions,
    build_clean_dataset,
    dedupe_by_participant_issue,
    export_issue_datasets,
    generate_prompt_stats,
    ensure_shared_schema,
    load_raw,
    map_rows_to_examples,
    parse_issue_repo_specs,
    save_dataset,
    validate_required_columns,
)
from clean_data._datasets import _ColumnUnionLoader, normalize_split_mappings
from clean_data.prompt.constants import REQUIRED_PROMPT_COLUMNS
from clean_data.filters import filter_prompt_ready

pytestmark = pytest.mark.clean_data


def _example_row() -> dict:
    slate_items = [
        {"title": "Option A", "id": "next_video"},
        {"title": "Option B", "id": "other"},
    ]
    history = [
        {"id": "current", "title": "Current", "watch_seconds": 12, "total_length": 20},
        {"id": "earlier", "title": "Earlier", "watch_seconds": 30, "total_length": 45},
    ]
    return {
        "issue": "minimum_wage",
        "slate_items_json": json.dumps(slate_items),
        "watched_detailed_json": json.dumps(history),
        "watched_vids_json": json.dumps(["earlier", "current", "next_video"]),
        "current_video_id": "current",
        "current_video_title": "Current",
        "next_video_id": "next_video",
        "n_options": 2,
        "viewer_profile_sentence": "",
    }


def test_validate_required_columns_detects_missing():
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": ["example"]})})
    with pytest.raises(ValueError):
        validate_required_columns(dataset)


def test_ensure_shared_schema_populates_missing_columns():
    left = Dataset.from_dict({"prompt": ["a"], "answer": ["1"]})
    right = Dataset.from_dict({"prompt": ["b"]})
    aligned = ensure_shared_schema({"train": left, "validation": right})
    assert set(aligned["validation"].column_names) == set(aligned["train"].column_names)


def test_build_clean_dataset_from_saved_directory():
    dataset = DatasetDict({"train": Dataset.from_dict({k: [v] for k, v in _example_row().items()})})
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "saved_dataset"
        dataset.save_to_disk(path)
        result = build_clean_dataset(str(path), options=BuildOptions(validation_ratio=0.0))
    assert "train" in result
    row = result["train"][0]
    assert row["prompt"]
    assert row["answer"] == "1"


def test_filter_prompt_ready_removes_invalid_rows():
    good = Dataset.from_dict({key: [value] for key, value in _example_row().items()})
    filtered = filter_prompt_ready(DatasetDict({"train": good}))
    assert len(filtered["train"]) == 1

    bad_example = _example_row()
    bad_example["slate_items_json"] = json.dumps([])
    bad = Dataset.from_dict({key: [value] for key, value in bad_example.items()})
    filtered_bad = filter_prompt_ready(DatasetDict({"train": bad}))
    assert len(filtered_bad["train"]) == 0


def test_load_raw_falls_back_to_column_union_when_schema_mismatches(monkeypatch: pytest.MonkeyPatch):
    sentinel = DatasetDict({"train": Dataset.from_dict({"prompt": ["a"]})})
    union_calls: Dict[str, Any] = {}

    def fake_load_dataset(dataset_name: str):
        raise clean_data.clean_data.DatasetGenerationCastError("mismatch")  # type: ignore[attr-defined]

    def fake_union(dataset_name: str):
        union_calls["name"] = dataset_name
        return sentinel

    monkeypatch.setattr(clean_data.clean_data.datasets, "load_dataset", fake_load_dataset)  # type: ignore[attr-defined]
    monkeypatch.setattr("clean_data.clean_data._load_dataset_with_column_union", fake_union)

    result = load_raw("stub/dataset")

    assert result is sentinel
    assert union_calls["name"] == "stub/dataset"


def test_load_raw_rejects_unsupported_extensions(tmp_path: Path):
    invalid = tmp_path / "data.xml"
    invalid.write_text("<root />", encoding="utf-8")

    with pytest.raises(ValueError):
        load_raw(str(invalid))


def test_map_rows_to_examples_applies_prompt_and_sol_key(monkeypatch: pytest.MonkeyPatch):
    dataset = DatasetDict({"train": Dataset.from_dict({"value": [1]})})

    captured: Dict[str, Any] = {}

    def fake_row_to_example(row: dict, prompt: str, sol_key: str, max_history: int) -> dict:
        captured.update({"prompt": prompt, "sol_key": sol_key, "max_history": max_history})
        return {"prompt": f"seen:{row['value']}", "answer": "ok"}

    monkeypatch.setattr("clean_data.clean_data.row_to_example", fake_row_to_example)

    mapped = map_rows_to_examples(
        dataset,
        system_prompt="system",
        sol_key="solution",
        max_history=7,
        num_proc=None,
    )

    assert mapped["train"][0]["prompt"] == "seen:1"
    assert captured == {"prompt": "system", "sol_key": "solution", "max_history": 7}


def test_map_rows_to_examples_validates_num_proc():
    dataset = DatasetDict({"train": Dataset.from_dict({"value": [1]})})
    with pytest.raises(ValueError):
        map_rows_to_examples(dataset, system_prompt=None, sol_key=None, max_history=1, num_proc=0)


def test_ensure_shared_schema_adds_missing_sequence_feature():
    template = Features({"seq": HFSequence(Value("string")), "prompt": Value("string")})
    left = Dataset.from_dict({"seq": [["a", "b"]], "prompt": ["x"]}).cast(template)
    right = Dataset.from_dict({"prompt": ["y"]})

    aligned = ensure_shared_schema({"train": left, "validation": right})

    assert aligned["validation"][0]["seq"] == []


def test_validate_required_columns_accepts_complete_dataset():
    row = {column: [""] for column in REQUIRED_PROMPT_COLUMNS}
    dataset = DatasetDict({"train": Dataset.from_dict(row)})
    validate_required_columns(dataset)  # should not raise


def test_build_clean_dataset_respects_custom_split_names(monkeypatch: pytest.MonkeyPatch):
    raw = DatasetDict(
        {
            "custom_train": Dataset.from_dict({"prompt": ["t"], "answer": ["1"]}),
            "custom_eval": Dataset.from_dict({"prompt": ["e"], "answer": ["2"]}),
        }
    )

    monkeypatch.setattr("clean_data.clean_data.load_raw", lambda name, validation_ratio: raw)
    monkeypatch.setattr("clean_data.clean_data.filter_prompt_ready", lambda ds, **_: ds)
    monkeypatch.setattr("clean_data.clean_data.compute_issue_counts", lambda ds: {})
    monkeypatch.setattr("clean_data.clean_data.map_rows_to_examples", lambda ds, **_: ds)
    monkeypatch.setattr("clean_data.clean_data.ensure_shared_schema", lambda ds: ds)

    recorded: Dict[str, DatasetDict] = {}

    def fake_validate(dataset: DatasetDict) -> None:
        recorded["final"] = dataset

    monkeypatch.setattr("clean_data.clean_data.validate_required_columns", fake_validate)

    options = BuildOptions(train_split="custom_train", validation_split="custom_eval")
    result = build_clean_dataset("stub", options=options)

    assert set(result.keys()) == {"train", "validation"}
    assert result["train"][0]["prompt"] == "t"
    assert result["validation"][0]["prompt"] == "e"
    assert recorded["final"]["train"][0]["prompt"] == "t"


def test_normalize_split_mappings_supports_strings_and_sequences():
    mapping = {"train": "train.csv", "validation": ("val1.csv", "val2.csv")}
    normalized = normalize_split_mappings(mapping)
    assert normalized == {"train": ["train.csv"], "validation": ["val1.csv", "val2.csv"]}


def test_normalize_split_mappings_rejects_unsupported_types():
    with pytest.raises(RuntimeError):
        normalize_split_mappings({"train": {"unexpected": "value"}})


class _MemoryBuffer(io.BytesIO):
    """Context-manager friendly BytesIO used by the fake filesystem."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class _InMemoryFS:
    """Minimal filesystem stub exposing the ``open`` interface used by the loader."""

    def __init__(self, files: dict[str, bytes]):
        self._files = files

    def open(self, path: str, mode: str = "rb") -> _MemoryBuffer:
        if "b" not in mode:
            raise ValueError("Only binary mode is supported.")
        try:
            data = self._files[path]
        except KeyError as err:
            raise FileNotFoundError(path) from err
        return _MemoryBuffer(data)


def _make_loader(files: dict[str, bytes] | None = None) -> _ColumnUnionLoader:
    """Helper to instantiate the column-union loader with a stub builder."""

    files = files or {"placeholder.csv": b""}
    fs = _InMemoryFS(files)
    builder = type("Builder", (), {"_fs": fs})()
    features = Features(
        {
            "prompt": Value("string"),
            "answer": Value("string"),
            "extra": Value("string"),
        }
    )
    return _ColumnUnionLoader("dataset", builder, features)


def test_column_union_loader_decompresses_common_archives():
    csv_payload = b"prompt,answer\nhello,1\n"

    gzip_payload = gzip.compress(csv_payload)
    lzma_payload = lzma.compress(csv_payload)
    bz2_payload = bz2.compress(csv_payload)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as archive:
        archive.writestr("data.csv", csv_payload)
    zip_payload = zip_buffer.getvalue()

    assert _ColumnUnionLoader._decompress_payload(gzip_payload, "data.csv.gz") == csv_payload
    assert _ColumnUnionLoader._decompress_payload(lzma_payload, "data.csv.xz") == csv_payload
    assert _ColumnUnionLoader._decompress_payload(bz2_payload, "data.csv.bz2") == csv_payload
    assert _ColumnUnionLoader._decompress_payload(zip_payload, "archive.zip") == csv_payload


def test_column_union_loader_canonicalises_pre_columns():
    loader = _make_loader()
    frame = loader._postprocess_frame(pd.DataFrame({"prompt_pre": ["hi"], "answer_pre": ["42"]}))
    assert set(frame.columns) == {"prompt", "answer"}


def test_column_union_loader_combines_frames_and_casts_types():
    loader = _make_loader()
    left = pd.DataFrame({"prompt": ["a"], "answer": ["1"]})
    right = pd.DataFrame({"prompt": ["b"], "extra_pre": ["2"]})
    combined = loader._combine_frames([left, loader._postprocess_frame(right)])
    assert list(combined.columns) == ["answer", "extra", "prompt"]
    # ensure missing columns are filled with pandas.NA
    assert pd.isna(combined.loc[1, "answer"])
    cast = loader._apply_feature_casts(combined.copy())
    assert cast["prompt"].dtype == pd.StringDtype()


def test_dedupe_by_participant_issue_removes_duplicates_and_preserves_schema():
    base = Dataset.from_dict(
        {
            "participant_id": ["p1", "p1", "p2"],
            "issue": ["i1", "i1", "i2"],
            "prompt": ["first", "duplicate", "second"],
        }
    )
    dataset = DatasetDict({"train": base})

    deduped = dedupe_by_participant_issue(dataset)

    assert len(deduped["train"]) == 2
    assert deduped["train"][0]["prompt"] == "first"
    assert deduped["train"].features == base.features


def test_dedupe_by_participant_issue_noop_when_keys_missing():
    original = Dataset.from_dict({"participant_id": ["p1"], "prompt": ["only"]})
    dataset = DatasetDict({"train": original})

    deduped = dedupe_by_participant_issue(dataset)

    assert len(deduped["train"]) == len(original)
    assert deduped["train"][0]["prompt"] == original[0]["prompt"]


def test_save_dataset_creates_directory_and_calls_save(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": ["x"]})})
    calls: Dict[str, Any] = {}

    def fake_save(path: str) -> None:
        calls["path"] = path

    monkeypatch.setattr(dataset, "save_to_disk", fake_save)
    output = tmp_path / "nested" / "outputs"

    save_dataset(dataset, output)

    assert output.exists()
    assert calls["path"] == str(output)


def test_generate_prompt_stats_invokes_report_builder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"prompt": ["train"]}),
            "validation": Dataset.from_dict({"prompt": ["val"]}),
        }
    )
    captured: Dict[str, Any] = {}

    def fake_builder(ds: DatasetDict, output_dir: Path, train_split: str, validation_split: str) -> None:
        captured["keys"] = sorted(ds.keys())
        captured["output_dir"] = Path(output_dir)
        captured["splits"] = (train_split, validation_split)

    monkeypatch.setattr("clean_data.clean_data._load_prompt_feature_report_builder", lambda: fake_builder)

    generate_prompt_stats(dataset, tmp_path, train_split="train", validation_split="validation")

    assert captured["keys"] == ["train", "validation"]
    assert captured["output_dir"] == tmp_path
    assert captured["splits"] == ("train", "validation")


def test_generate_prompt_stats_raises_when_builder_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"prompt": ["train"]}),
            "validation": Dataset.from_dict({"prompt": ["val"]}),
        }
    )
    monkeypatch.setattr("clean_data.clean_data._load_prompt_feature_report_builder", lambda: None)

    with pytest.raises(ImportError):
        generate_prompt_stats(dataset, tmp_path)


def test_export_issue_datasets_saves_each_issue(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = DatasetDict(
        {
            "train": Dataset.from_dict({"issue": ["alpha", "beta"], "prompt": ["t1", "t2"]}),
            "validation": Dataset.from_dict({"issue": ["beta"], "prompt": ["v1"]}),
        }
    )

    save_calls: list[Path] = []
    push_calls: list[tuple[str, str | None]] = []

    def fake_save(self: DatasetDict, path: str) -> None:
        save_calls.append(Path(path))

    def fake_push(self: DatasetDict, repo_id: str, token: str | None = None) -> None:
        push_calls.append((repo_id, token))

    monkeypatch.setattr(DatasetDict, "save_to_disk", fake_save, raising=False)
    monkeypatch.setattr(DatasetDict, "push_to_hub", fake_push, raising=False)

    export_issue_datasets(
        dataset,
        tmp_path,
        issue_repo_map={"alpha": "repo/alpha"},
        push_to_hub=True,
        hub_token="secret",
    )

    saved_dirs = {path.name for path in save_calls}
    assert saved_dirs == {"alpha", "beta"}
    assert push_calls == [("repo/alpha", "secret")]


def test_export_issue_datasets_noop_without_issue_column(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    dataset = DatasetDict({"train": Dataset.from_dict({"prompt": ["only"]})})

    def sentinel_save(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("save_to_disk should not be called")

    monkeypatch.setattr(DatasetDict, "save_to_disk", sentinel_save, raising=False)

    export_issue_datasets(dataset, tmp_path, issue_repo_map={"alpha": "repo/alpha"}, push_to_hub=False)


def test_build_clean_dataset_uses_first_split_when_train_missing(monkeypatch: pytest.MonkeyPatch):
    raw = DatasetDict({"other": Dataset.from_dict({"prompt": ["x"], "answer": ["1"]})})

    monkeypatch.setattr("clean_data.clean_data.load_raw", lambda name, validation_ratio: raw)
    monkeypatch.setattr("clean_data.clean_data.filter_prompt_ready", lambda ds, **_: ds)
    monkeypatch.setattr("clean_data.clean_data.compute_issue_counts", lambda ds: {})
    monkeypatch.setattr("clean_data.clean_data.map_rows_to_examples", lambda ds, **_: ds)
    monkeypatch.setattr("clean_data.clean_data.ensure_shared_schema", lambda ds: ds)
    monkeypatch.setattr("clean_data.clean_data.validate_required_columns", lambda ds: None)

    result = build_clean_dataset("stub")

    assert list(result.keys()) == ["train"]
    assert result["train"][0]["prompt"] == "x"


def test_parse_issue_repo_specs_parses_assignments():
    mapping = parse_issue_repo_specs(["alpha = repo/alpha", "beta=repo/beta"])
    assert mapping == {"alpha": "repo/alpha", "beta": "repo/beta"}


def test_parse_issue_repo_specs_rejects_bad_format():
    with pytest.raises(ValueError):
        parse_issue_repo_specs(["invalid-format"])

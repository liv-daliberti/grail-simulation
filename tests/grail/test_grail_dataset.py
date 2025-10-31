"""Unit tests for dataset preparation helpers in ``grail.grail_dataset``."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List

import pytest

if "trl" not in sys.modules:  # pragma: no cover - optional dependency stub
    trl_stub = types.ModuleType("trl")
    trl_stub.ModelConfig = type("ModelConfig", (), {})
    trl_stub.ScriptArguments = type("ScriptArguments", (), {})
    trl_stub.GRPOConfig = type("GRPOConfig", (), {})
    trl_stub.SFTConfig = type("SFTConfig", (), {})
    sys.modules["trl"] = trl_stub

from grail import grail_dataset as module


@dataclass
class _FakeSplit:
    """Minimal stand-in for ``datasets.Dataset`` with only the features we need."""

    name: str
    rows: List[Dict[str, Any]]
    column_names: List[str] = field(init=False)

    def __post_init__(self) -> None:
        keys: set[str] = set()
        for row in self.rows:
            keys.update(row.keys())
        self.column_names = sorted(keys)

    def remove_columns(self, drop: Iterable[str]) -> "_FakeSplit":
        drop_set = set(drop)
        return _FakeSplit(
            self.name,
            [{k: v for k, v in row.items() if k not in drop_set} for row in self.rows],
        )


class _FakeDatasetDict(dict):
    """Lightweight ``DatasetDict`` drop-in that supports ``filter`` and ``map``."""

    def filter(self, fn, fn_kwargs: Dict[str, Any] | None = None):
        kwargs = fn_kwargs or {}
        filtered = {}
        for name, split in self.items():
            kept_rows = [row for row in split.rows if fn(row, **kwargs)]
            filtered[name] = _FakeSplit(name, kept_rows)
        return _FakeDatasetDict(filtered)

    def map(self, fn, load_from_cache_file: bool = False):  # noqa: ARG002 - parity
        mapped = {}
        for name, split in self.items():
            new_rows = []
            for row in split.rows:
                result = fn(row)
                if result is not None:
                    new_rows.append(result)
            mapped[name] = _FakeSplit(name, new_rows)
        return _FakeDatasetDict(mapped)


def test_prepare_dataset_formats_rows_and_drops_extras(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_prepare_dataset`` should filter, format, and prune dataset columns."""

    converted = ["converted-items"]

    monkeypatch.setattr(module, "as_list_json", lambda payload: converted if payload else [])
    monkeypatch.setattr(module, "collect_passthrough_fields", lambda example: {"viewer": example["viewer"]})

    drop_calls: list[tuple[object, str]] = []
    monkeypatch.setattr(module, "drop_marked_rows", lambda data, split: drop_calls.append((data, split)))

    def fake_validator(example: Dict[str, Any], *, solution_key: str | None) -> bool:  # noqa: D401 - helper
        return not example.get("drop_me")

    monkeypatch.setattr(module, "make_slate_validator", lambda **_: fake_validator)

    def fake_call_row_to_training_example(
        example: Dict[str, Any],
        *,
        extra_fields_fn,
        **_: Any,
    ) -> Dict[str, Any]:
        formatted = {
            "prompt": f"prompt-{example['id']}",
            "response": "ok",
            "unused_column": "remove-me",
        }
        formatted.update(extra_fields_fn(example, []))
        return formatted

    monkeypatch.setattr(module, "call_row_to_training_example", fake_call_row_to_training_example)
    monkeypatch.setattr(
        module,
        "TRAIN_KEEP_COLUMNS",
        {"prompt", "response", "slate_items_with_meta"},
    )

    raw = _FakeDatasetDict(
        {
            "train": _FakeSplit(
                "train",
                [
                    {"id": 1, "viewer": "viewer-1", "slate_items_json": "[1, 2, 3]", "drop_me": False},
                    {"id": 2, "viewer": "viewer-2", "slate_items_json": "[4, 5]", "drop_me": True},
                ],
            )
        }
    )

    prepared = module._prepare_dataset(
        raw_dataset=raw,
        system_prompt="sys",
        solution_key="gold",
        max_hist=3,
        train_split="train",
    )

    assert drop_calls and drop_calls[0][1] == "train"

    train_split = prepared["train"]
    assert isinstance(train_split, _FakeSplit)
    assert train_split.column_names == ["prompt", "response", "slate_items_with_meta"]
    assert len(train_split.rows) == 1, "Filtered dataset should retain only valid rows"

    row = train_split.rows[0]
    assert "unused_column" not in row
    assert row["slate_items_with_meta"] == converted


def test_build_dataset_and_tokenizer_respects_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """``_build_dataset_and_tokenizer`` should honour ``GRAIL_MAX_HISTORY``."""

    dataset = object()
    tokenizer = object()

    call_args: dict[str, Any] = {}
    script_args = types.SimpleNamespace(dataset_train_split="train", dataset_solution_column="answer")
    training_args = types.SimpleNamespace(system_prompt="hello")
    model_args = types.SimpleNamespace(model="stub")

    monkeypatch.setattr(module, "get_dataset", lambda args: dataset if args is script_args else None)
    monkeypatch.setattr(
        module,
        "get_tokenizer",
        lambda model_args_param, training_args_param: (model_args_param, training_args_param),
    )

    def fake_prepare(raw_dataset, system_prompt, solution_key, max_hist, train_split):
        call_args.update(
            {
                "raw_dataset": raw_dataset,
                "system_prompt": system_prompt,
                "solution_key": solution_key,
                "max_hist": max_hist,
                "train_split": train_split,
            }
        )
        return {"train": []}

    monkeypatch.setattr(module, "_prepare_dataset", fake_prepare)
    monkeypatch.setenv("GRAIL_MAX_HISTORY", "42")

    returned_dataset, returned_tokenizer = module._build_dataset_and_tokenizer(
        script_args,
        training_args,
        model_args,
    )

    assert returned_dataset == {"train": []}
    assert returned_tokenizer == (model_args, training_args)
    assert call_args["raw_dataset"] is dataset
    assert call_args["system_prompt"] == "hello"
    assert call_args["solution_key"] == "answer"
    assert call_args["max_hist"] == 42
    assert call_args["train_split"] == "train"


def test_build_dataset_and_tokenizer_raises_for_invalid_history(monkeypatch: pytest.MonkeyPatch) -> None:
    script_args = types.SimpleNamespace(dataset_train_split="train", dataset_solution_column=None)
    training_args = types.SimpleNamespace(system_prompt=None)
    model_args = object()

    monkeypatch.setattr(module, "get_dataset", lambda *_: {"train": []}, raising=False)
    monkeypatch.setattr(module, "get_tokenizer", lambda *_: object(), raising=False)
    monkeypatch.setattr(module, "_prepare_dataset", lambda *_: {"train": []}, raising=False)
    monkeypatch.setenv("GRAIL_MAX_HISTORY", "not-an-int")

    with pytest.raises(ValueError):
        module._build_dataset_and_tokenizer(script_args, training_args, model_args)


def test_prepare_dataset_drops_invalid_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _FakeDatasetDict(
        {
            "train": _FakeSplit(
                "train",
                [
                    {"id": 1, "viewer": "valid", "slate_items_json": "[1]", "drop_me": False},
                    {"id": 2, "viewer": "invalid", "slate_items_json": "[2]", "drop_me": False},
                ],
            )
        }
    )

    monkeypatch.setattr(module, "collect_passthrough_fields", lambda example: {"viewer": example["viewer"]})
    monkeypatch.setattr(module, "as_list_json", lambda payload: ["ok"] if payload else [])
    monkeypatch.setattr(module, "drop_marked_rows", lambda *_: None)
    monkeypatch.setattr(
        module,
        "make_slate_validator",
        lambda **_: lambda example, **__: example["id"] == 1,
    )

    def fake_call_row_to_training_example(example, **_kwargs):
        return {"prompt": f"id-{example['id']}", "response": "ok"} if example["id"] == 1 else None

    monkeypatch.setattr(module, "call_row_to_training_example", fake_call_row_to_training_example)
    monkeypatch.setattr(module, "TRAIN_KEEP_COLUMNS", {"prompt", "response"})

    prepared = module._prepare_dataset(
        raw_dataset=raw,
        system_prompt=None,
        solution_key=None,
        max_hist=3,
        train_split="train",
    )

    train_split = prepared["train"]
    assert isinstance(train_split, _FakeSplit)
    assert len(train_split.rows) == 1
    assert train_split.rows[0]["prompt"] == "id-1"

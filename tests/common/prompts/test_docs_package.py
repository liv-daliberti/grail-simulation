"""Integration tests for the refactored ``common.prompts.docs`` package."""

from __future__ import annotations

import pytest

from common.prompts import docs as docs_pkg
from common.prompts.docs.builder import PromptDocumentBuilder
from common.prompts.docs.extra_fields import format_extra_field
from common.prompts.docs.slate import extract_slate_items
from prompt_builder.constants import GUN_FIELD_LABELS, MIN_WAGE_FIELD_LABELS


def test_merge_default_extra_fields_deduplicates_and_preserves_order() -> None:
    result = docs_pkg.merge_default_extra_fields(
        ["viewer_profile", "custom", "state_text", "custom"]
    )
    assert result[:2] == docs_pkg.DEFAULT_EXTRA_TEXT_FIELDS
    assert result[-1] == "custom"
    assert len(result) == 3


def test_merge_default_extra_fields_ignores_blank_tokens() -> None:
    result = docs_pkg.merge_default_extra_fields(
        ["", "   ", None, "viewer_profile  ", "Custom", "custom", "\nstate_text\n"]
    )
    assert result[:2] == docs_pkg.DEFAULT_EXTRA_TEXT_FIELDS
    assert "" not in result
    assert result.count("state_text") == 1
    assert result.count("viewer_profile") == 1
    assert "Custom" in result


def test_extract_slate_items_prefers_structured_metadata_and_deduplicates() -> None:
    example = {
        "slate_items": [
            {"title": "First", "video_id": "abc123def45"},
            {"title": "Second", "video_id": "abc123def45"},
            {"title": "", "video_id": "xyz98765432"},
        ],
        "trajectory_json": {"order": [{"title": "Fallback", "video_id": "xyz98765432"}]},
    }

    seen = extract_slate_items(example, lambda video_id: f"title:{video_id}")
    assert seen == [
        ("First", "abc123def45"),
        ("title:xyz98765432", "xyz98765432"),
    ]


def test_format_extra_field_recovers_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: f"{field}:{value}",
    )
    formatted = format_extra_field({"pid1": "independent"}, "pid1")
    assert formatted == "Party identification: pid1:independent"


@pytest.mark.parametrize(
    "field_name,label",
    sorted(docs_pkg.EXTRA_FIELD_LABELS.items(), key=lambda item: item[0]),
)
def test_format_extra_field_uses_all_known_labels(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    label: str,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "formatted",
    )
    formatted = format_extra_field({field_name: "value"}, field_name)
    assert formatted == f"{label}: formatted"


@pytest.mark.parametrize(
    "field_name,label",
    sorted(MIN_WAGE_FIELD_LABELS.items(), key=lambda item: item[0]),
)
def test_format_extra_field_includes_min_wage_labels(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    label: str,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "formatted",
    )
    formatted = format_extra_field({field_name: "value"}, field_name)
    assert formatted == f"{label}: formatted"


@pytest.mark.parametrize(
    "field_name,label",
    sorted(GUN_FIELD_LABELS.items(), key=lambda item: item[0]),
)
def test_format_extra_field_includes_gun_labels(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    label: str,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "formatted",
    )
    formatted = format_extra_field({field_name: "value"}, field_name)
    assert formatted == f"{label}: formatted"


def test_format_extra_field_returns_blank_when_value_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "",
    )
    formatted = format_extra_field({"pid1": None}, "pid1")
    assert formatted == ""


def test_format_extra_field_pretty_prints_unknown_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "formatted",
    )
    formatted = format_extra_field({"custom_field": "raw"}, "custom_field")
    assert formatted == "Custom field: formatted"


@pytest.mark.parametrize(
    "formatted_value,expected_fragment",
    [
        (123, "123"),
        (["A", "B"], "['A', 'B']"),
        ("Already Formatted", "Already Formatted"),
    ],
)
def test_format_extra_field_handles_complex_formatted_values(
    monkeypatch: pytest.MonkeyPatch,
    formatted_value,
    expected_fragment: str,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value, _fv=formatted_value: _fv,
    )
    result = format_extra_field({"pid2": "raw"}, "pid2")
    assert result == f"Party lean: {expected_fragment}"


def test_format_extra_field_normalizes_child18_yes_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "No children in household",
    )
    formatted_no = format_extra_field(
        {"child18": "No children in household"}, "child18"
    )
    assert formatted_no == "Children in household: no"

    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: "Two children",
    )
    formatted_yes = format_extra_field({"child18": "Two children"}, "child18")
    assert formatted_yes == "Children in household: yes"


def test_prompt_document_builder_prepare_training_documents_includes_extras(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build_user_prompt(example, max_hist):
        raise ValueError("force fallback")

    monkeypatch.setattr(
        "common.prompts.docs.builder.build_user_prompt",
        fake_build_user_prompt,
    )
    monkeypatch.setattr(
        "prompt_builder.value_maps.format_field_value",
        lambda field, value: f"{field}:{value}",
    )

    builder = PromptDocumentBuilder(
        prompt_column="prompt_col",
        solution_column="solution",
        max_history=3,
        title_lookup=lambda video_id: f"title:{video_id}",
        log_prefix="test",
    )

    class _Dataset:
        def __init__(self):
            self.features = {"prompt_col": "text", "solution": "label"}
            self._rows = [
                {
                    "prompt": "",
                    "prompt_col": "existing prompt",
                    "viewer_profile": "profile text",
                    "slate_items": [{"title": "Slate Title", "video_id": "abc123def45"}],
                    "solution": "abc123def45",
                    "pid1": "independent",
                    "ideo1": "moderate",
                    "custom_field": "custom value",
                }
            ]

        def __len__(self) -> int:
            return len(self._rows)

        def __getitem__(self, index: int) -> dict:
            return self._rows[index]

    dataset = _Dataset()

    docs, ids, titles = builder.prepare_training_documents(
        dataset,
        max_train=1,
        seed=1,
        extra_fields=["pid1", "ideo1", "custom_field"],
    )

    assert ids == ["abc123def45"]
    assert titles == ["title:abc123def45"]
    assert "existing prompt" in docs[0]
    assert "Slate Title" in docs[0]
    assert docs[0].count("Party identification: pid1:independent") == 1
    assert docs[0].count("Political ideology: ideo1:moderate") == 1
    assert docs[0].count("Custom field: custom_field:custom value") == 1
    first_index = docs[0].index("Party identification: pid1:independent")
    second_index = docs[0].index("Political ideology: ideo1:moderate")
    third_index = docs[0].index("Custom field: custom_field:custom value")
    assert first_index < second_index < third_index

"""Focused tests for lightweight helpers in :mod:`gpt4o.evaluate`."""

from __future__ import annotations

from pathlib import Path

import pytest

from gpt4o import evaluate
from tests.helpers.datasets_stub import ensure_datasets_stub

pytestmark = pytest.mark.gpt4o

ensure_datasets_stub()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<answer>\n 2 \n</answer>", 2),
        ("Reasoning\n<answer>5</answer>\n", 5),
    ],
)
def test_parse_index_from_output_prefers_answer_tag(raw: str, expected: int) -> None:
    """Indexes embedded inside <answer> tags should be extracted exactly."""

    assert evaluate._parse_index_from_output(raw) == expected  # pylint: disable=protected-access


def test_parse_index_from_output_falls_back_to_trailing_lines() -> None:
    """When tags are missing, fallback parsing should inspect trailing lines."""

    raw = "some text\nmaybe numbers\n3\n"
    assert evaluate._parse_index_from_output(raw) == 3  # pylint: disable=protected-access


def test_parse_index_from_output_returns_none_for_invalid_output() -> None:
    """Non-numeric responses should not raise and instead yield ``None``."""

    assert evaluate._parse_index_from_output("no selection provided") is None  # pylint: disable=protected-access


def test_ensure_output_dir_creates_missing_directory(tmp_path: Path) -> None:
    """Creating a new output directory should succeed with overwrite enabled."""

    target = tmp_path / "outputs"
    evaluate._ensure_output_dir(target, overwrite=True)  # pylint: disable=protected-access
    assert target.exists()


def test_ensure_output_dir_raises_when_exists_and_no_overwrite(tmp_path: Path) -> None:
    """Existing directories should raise when overwrite is disabled."""

    target = tmp_path / "existing"
    target.mkdir()
    with pytest.raises(FileExistsError):
        evaluate._ensure_output_dir(target, overwrite=False)  # pylint: disable=protected-access


def test_evaluation_limits_from_arg_sets_target() -> None:
    """``EvaluationLimits`` should expose both capped and uncapped forms."""

    capped = evaluate.EvaluationLimits.from_arg(25)
    assert capped.eval_max == 25
    assert capped.target == 25

    uncapped = evaluate.EvaluationLimits.from_arg(0)
    assert uncapped.eval_max == 0
    assert uncapped.target is None


def test_parse_filter_strips_tokens_and_handles_all_keyword() -> None:
    """Comma-separated filters should normalise tokens and clear on ``all``."""

    tokens, lowered = evaluate.EvaluationRunner._parse_filter(" Issue1 , issue2 , all ")  # pylint: disable=protected-access
    assert tokens == ["Issue1", "issue2", "all"]
    assert lowered == set()

    tokens, lowered = evaluate.EvaluationRunner._parse_filter("alpha, beta")  # pylint: disable=protected-access
    assert tokens == ["alpha", "beta"]
    assert lowered == {"alpha", "beta"}

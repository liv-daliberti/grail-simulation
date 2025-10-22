"""Tests for opinion-stage participant collapsing."""

from __future__ import annotations

import pytest

from xgb import opinion


pytestmark = pytest.mark.xgb


def test_collect_examples_prefers_latest_step(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure phase II opinion rows collapse to the final watch step per participant."""

    def fake_assemble_document(row: dict, extra_fields) -> str:  # pragma: no cover - simple stub
        return row["doc"]

    monkeypatch.setattr(opinion, "assemble_document", fake_assemble_document)

    spec = opinion.OpinionSpec(
        key="study1",
        issue="gun_control",
        label="Study 1",
        before_column="before",
        after_column="after",
    )
    dataset = [
        {
            "issue": "gun_control",
            "participant_study": "study1",
            "participant_id": "alpha",
            "before": 0.1,
            "after": 0.2,
            "doc": "early alpha",
            "step_index": 0,
        },
        {
            "issue": "gun_control",
            "participant_study": "study1",
            "participant_id": "alpha",
            "before": 0.1,
            "after": 0.4,
            "doc": "late alpha",
            "step_index": 5,
        },
        {
            "issue": "gun_control",
            "participant_study": "study1",
            "participant_id": "beta",
            "before": 0.3,
            "after": 0.35,
            "doc": "missing step beta",
        },
        {
            "issue": "gun_control",
            "participant_study": "study1",
            "participant_id": "beta",
            "before": 0.3,
            "after": 0.5,
            "doc": "late beta",
            "step_index": 2,
        },
    ]

    examples = opinion.collect_examples(
        dataset,
        spec=spec,
        extra_fields=(),
        max_participants=0,
        seed=0,
    )
    assert len(examples) == 2

    results = {ex.participant_id: ex for ex in examples}
    assert results["alpha"].after == pytest.approx(0.4)
    assert results["alpha"].document == "late alpha"
    assert results["beta"].after == pytest.approx(0.5)
    assert results["beta"].document == "late beta"

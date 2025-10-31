#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

import json
from pathlib import Path

import pytest

from grpo.next_video import (
    FilterSelection,
    NextVideoDatasetSpec,
    NextVideoEvaluationLimits,
    NextVideoEvaluationSettings,
    NextVideoPromptSettings,
    PreparedExample,
    run_next_video_evaluation,
)


def _seed_prediction_row(path: Path) -> None:
    row = {
        "messages": [{"role": "user", "content": "Q?"}],
        "gpt_output": "<answer>1</answer>",
        "parsed_index": 1,
        "gold_index": 1,
        "n_options": 2,
        "correct": True,
        "eligible": True,
        "issue": "gun_control",
        "participant_study": "study1",
        "position_index": 0,
        "position_bucket": "1",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


@pytest.mark.parametrize("overwrite", [False])
def test_next_video_resume_appends_and_respects_cap(tmp_path: Path, monkeypatch, overwrite: bool) -> None:
    # Prepare run directory with a single saved prediction.
    out_dir = tmp_path / "models" / "grpo"
    label = "unit"
    run_dir = out_dir / "next_video" / label
    preds_path = run_dir / "predictions.jsonl"
    _seed_prediction_row(preds_path)

    # Monkeypatch dataset loader to avoid HF datasets; return two trivial rows.
    from grpo import next_video as nv

    def fake_load_dataset_split(*_args, **_kwargs):
        return [
            {"issue": "gun_control", "participant_study": "study1", "position_index": 0},
            {"issue": "gun_control", "participant_study": "study1", "position_index": 0},
        ]

    monkeypatch.setattr(nv, "load_dataset_split", fake_load_dataset_split)

    # Monkeypatch example preparation to produce 2 examples with gold_index=1.
    def fake_prepare_examples(rows, system_prompt, solution_key, max_history):  # noqa: ARG001
        ex = PreparedExample(
            messages=[{"role": "user", "content": "Q?"}],
            gold_index=1,
            gold_id="vid1",
            n_options=2,
            raw_row=rows[0],
        )
        ex2 = PreparedExample(
            messages=[{"role": "user", "content": "Q?"}],
            gold_index=1,
            gold_id="vid2",
            n_options=2,
            raw_row=rows[1],
        )
        return iter([ex, ex2])

    monkeypatch.setattr(nv, "prepare_examples", fake_prepare_examples)

    # Always produce the correct answer quickly.
    monkeypatch.setattr(nv, "generate_chat_completion", lambda *a, **k: "<answer>1</answer>")

    # Build evaluation settings with a cap of 2 total examples.
    settings = NextVideoEvaluationSettings(
        model_label=label,
        dataset=NextVideoDatasetSpec(name=str(tmp_path / "dataset"), split="validation", cache_dir=None),
        prompts=NextVideoPromptSettings(system_prompt="sys", solution_key=None, max_history=1),
        limits=NextVideoEvaluationLimits(max_examples=2),
        overwrite=overwrite,
        generation=type("Gen", (), {"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0})(),
        filters=FilterSelection.from_raw(issues=None, studies=None),
    )

    # Tokenizer/model placeholders (not used due to monkeypatch).
    tokenizer = object()
    model = object()

    result = run_next_video_evaluation(
        tokenizer=tokenizer,
        model=model,
        settings=settings,
        config_label=label,
        out_dir=out_dir / "next_video",
    )

    # Resume should have appended exactly one example and computed metrics over both.
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    if "metrics" in metrics:
        metrics = metrics["metrics"]
    assert metrics.get("n_total") == 2
    assert metrics.get("n_eligible") == 2
    assert pytest.approx(metrics.get("accuracy_overall"), rel=1e-6) == 1.0


#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

import json
from pathlib import Path

import grail.pipeline as grail_pipeline
import grpo.pipeline as grpo_pipeline


def _write_next_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "issue": "gun_control",
            "participant_study": "study1",
            "n_options": 3,
            "position_index": 0,
            "gold_index": 1,
            "parsed_index": 1,
            "eligible": True,
            "correct": True,
            "gpt_output": "<answer>1</answer>",
        },
        {
            "issue": "minimum_wage",
            "participant_study": "study2",
            "n_options": 2,
            "position_index": 1,
            "gold_index": 2,
            "parsed_index": 2,
            "eligible": True,
            "correct": True,
            "gpt_output": "<answer>2</answer>",
        },
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_opinion_predictions(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {"before": 2.0, "after": 3.0, "prediction": 3.0},
        {"before": 6.0, "after": 5.0, "prediction": 5.0},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_grail_reports_stage_rebuilds_from_partial_artifacts(tmp_path: Path, monkeypatch) -> None:
    label = "unit"
    out_dir = tmp_path / "models" / "grail"
    next_dir = out_dir / "next_video" / label
    opinion_dir = out_dir / "opinion" / label / "study1"
    _write_next_predictions(next_dir / "predictions.jsonl")
    _write_opinion_predictions(opinion_dir / "predictions.jsonl")

    # Ensure repo_root resolves to tmp_path for reports output
    import grpo.config as grpo_config
    monkeypatch.setattr(grpo_config, "REPO_ROOT", tmp_path)

    # Run only the reports stage; it should rebuild metrics.json files and render markdown.
    grail_pipeline.main(
        [
            "--stage",
            "reports",
            "--out-dir",
            str(out_dir),
            "--label",
            label,
            "--dataset",
            str(tmp_path / "dataset"),
        ]
    )

    # Check that metrics were created from predictions and reports were rendered.
    assert (next_dir / "metrics.json").exists(), "Next-video metrics should be rebuilt"
    assert (opinion_dir / "metrics.json").exists(), "Opinion study metrics should be rebuilt"

    reports_root = tmp_path / "reports" / "grail"
    assert (reports_root / "README.md").exists(), "Catalog should be present"
    assert (reports_root / "next_video" / "README.md").exists()
    assert (reports_root / "opinion" / "README.md").exists()


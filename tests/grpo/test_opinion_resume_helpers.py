#!/usr/bin/env python
# pylint: disable=missing-function-docstring
from __future__ import annotations

import json
from pathlib import Path

from common.opinion import DEFAULT_SPECS
from grpo.opinion_io import (
    _attempt_reuse_cached_result,
    _prepare_study_files,
    _resume_from_predictions_if_needed,
    _seed_accumulator_from_predictions,
    _write_predictions,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_seed_accumulator_from_predictions_filters_and_counts(tmp_path: Path) -> None:
    preds = tmp_path / "predictions.jsonl"
    rows = [
        {"before": "3.0", "after": 4.0, "prediction": "4.0", "messages": [], "raw_output": "ok"},
        {"before": "nan", "after": 5.0, "prediction": 4.0},  # skipped
        {"before": "oops", "after": 5.0, "prediction": 4.0},  # skipped
        {"before": 2.0, "after": 1.0, "prediction": 1.0, "messages": [], "raw_output": "ok"},
    ]
    _write_jsonl(preds, rows)

    acc = _seed_accumulator_from_predictions(preds)
    # 'nan' parses as float('nan') and is included by the loader
    assert acc.participants == 3
    assert len(acc.qa_entries) == 3
    # Use isnan checks to avoid equality pitfalls
    import math
    assert acc.truth_before[0] == 3.0 and math.isnan(acc.truth_before[1]) and acc.truth_before[2] == 2.0
    assert acc.truth_after == [4.0, 5.0, 1.0]
    assert acc.pred_after == [4.0, 4.0, 1.0]


def test_attempt_reuse_cached_result_reads_metrics_and_seeds(tmp_path: Path) -> None:
    out_dir = tmp_path / "opinion"
    spec = DEFAULT_SPECS[0]
    files = _prepare_study_files(out_dir, spec)

    # seed predictions
    _write_jsonl(files.predictions, [{"before": 3.0, "after": 4.0, "prediction": 4.0}])

    # write metrics with participants=5 (should override acc.participants)
    files.metrics.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": {"eligible": 1, "participants": 5}, "baseline": {}}
    files.metrics.write_text(json.dumps(payload), encoding="utf-8")

    reused = _attempt_reuse_cached_result(spec=spec, files=files, overwrite=False)
    assert reused is not None
    result, acc = reused
    assert result.participants == 5
    assert acc.participants == 1


def test_resume_from_predictions_if_needed_returns_seeded_and_count(tmp_path: Path) -> None:
    out_dir = tmp_path / "opinion"
    spec = DEFAULT_SPECS[0]
    files = _prepare_study_files(out_dir, spec)

    # write predictions only; no metrics
    _write_jsonl(files.predictions, [
        {"before": 1.0, "after": 2.0, "prediction": 2.0},
        {"before": 3.0, "after": 2.0, "prediction": 2.0},
    ])

    acc, processed = _resume_from_predictions_if_needed(files=files, spec=spec)
    assert processed == 2
    assert acc.participants == 2


def test_write_predictions_serializes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "predictions.jsonl"
    rows = [{"before": 1.0, "after": 2.0, "prediction": 2.0}]
    _write_predictions(path, rows)
    text = path.read_text(encoding="utf-8").strip()
    assert text == json.dumps(rows[0])

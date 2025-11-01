#!/usr/bin/env python
"""File I/O helpers for GRPO opinion evaluation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from common.pipeline.io import (
    write_metrics_json,
    write_segmented_markdown_log,
    iter_jsonl_rows,
    write_jsonl_rows,
)
from common.opinion import OpinionSpec

from .opinion_types import (
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
    _StudyAccumulator,
)

LOGGER = logging.getLogger("grpo.opinion")


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Persist opinion predictions to JSONL."""
    write_jsonl_rows(path, rows, ensure_ascii=False)


def _prepare_study_files(out_dir: Path, spec: OpinionSpec) -> OpinionStudyFiles:
    """Return filesystem paths for a study evaluation."""

    study_dir = out_dir / spec.key
    study_dir.mkdir(parents=True, exist_ok=True)
    return OpinionStudyFiles(
        metrics=study_dir / "metrics.json",
        predictions=study_dir / "predictions.jsonl",
        qa_log=study_dir / "qa.log",
    )


def _persist_study_outputs(
    files: OpinionStudyFiles,
    *,
    metrics: Mapping[str, object],
    baseline: Mapping[str, object],
    accumulator: _StudyAccumulator,
) -> None:
    """Write per-study evaluation artefacts to disk."""

    write_metrics_json(files.metrics, {"metrics": metrics, "baseline": baseline})
    _write_predictions(files.predictions, accumulator.predictions)
    write_segmented_markdown_log(
        files.qa_log,
        title="GRPO Opinion QA Log",
        entries=accumulator.qa_entries,
    )


def _seed_accumulator_from_predictions(path: Path) -> _StudyAccumulator:
    """Return an accumulator pre-populated from an existing predictions JSONL."""
    acc = _StudyAccumulator()
    if not path.exists():
        return acc
    for row in iter_jsonl_rows(path):
        if not isinstance(row, Mapping):  # defensive; iter_jsonl_rows enforces Mapping
            continue
        before = row.get("before")
        after = row.get("after")
        pred = row.get("prediction")
        try:
            before_value = float(before)
            after_value = float(after)
            prediction_value = float(pred)
        except (TypeError, ValueError):
            continue
        messages = row.get("messages")
        raw_output = str(row.get("raw_output") or "")
        prompt_dump = (
            json.dumps(messages, indent=2, ensure_ascii=False)
            if messages is not None
            else ""
        )
        qa_entry = "\n".join(
            [
                f"## Participant {row.get('participant_id', '<unknown>')}",
                f"- Study: {row.get('study', '<unknown>')}",
                f"- Before: {before_value:.2f}",
                f"- After: {after_value:.2f}",
                "",
                "### Prompt",
                prompt_dump,
                "",
                "### Model output",
                raw_output.strip(),
                "",
                f"### Parsed prediction: {prediction_value:.3f}",
            ]
        )
        # Mirror what .record would append.
        acc.predictions.append(row)
        acc.qa_entries.append(qa_entry)
        acc.truth_before.append(before_value)
        acc.truth_after.append(after_value)
        acc.pred_after.append(prediction_value)
    return acc


def _attempt_reuse_cached_result(
    *,
    spec: OpinionSpec,
    files: OpinionStudyFiles,
    overwrite: bool,
) -> Tuple[OpinionStudyResult, _StudyAccumulator] | None:
    """Return cached result when metrics exist and overwrite is disabled.

    Seeds combined vectors from predictions.jsonl for aggregation even when
    metrics are reused.
    """

    if overwrite or not files.metrics.exists():
        return None

    try:
        payload = json.loads(files.metrics.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = {"metrics": {}, "baseline": {}}
    metrics = payload.get("metrics", {}) or {}
    baseline = payload.get("baseline", {}) or {}
    acc = _seed_accumulator_from_predictions(files.predictions)
    summary = OpinionStudySummary(
        metrics=metrics,
        baseline=baseline,
        participants=int(metrics.get("participants") or acc.participants),
        eligible=int(metrics.get("eligible", 0)),
    )
    return OpinionStudyResult(study=spec, files=files, summary=summary), acc


def _resume_from_predictions_if_needed(
    *, files: OpinionStudyFiles, spec: OpinionSpec
) -> tuple[_StudyAccumulator, int]:
    """Resume partial evaluation from predictions when metrics are missing."""

    if files.predictions.exists() and not files.metrics.exists():
        seeded = _seed_accumulator_from_predictions(files.predictions)
        if seeded.participants:
            LOGGER.info(
                "[OPINION] resuming study=%s at %d processed participants (resume=true)",
                spec.key,
                seeded.participants,
            )
            print(
                (
                    f"[grpo.opinion] resume=true study={spec.key} "
                    f"processed={seeded.participants}"
                ),
                flush=True,
            )
        return seeded, seeded.participants
    return _StudyAccumulator(), 0


__all__ = [
    "_attempt_reuse_cached_result",
    "_prepare_study_files",
    "_persist_study_outputs",
    "_resume_from_predictions_if_needed",
    "_seed_accumulator_from_predictions",
    "_write_predictions",
]

#!/usr/bin/env python
"""Cache loaders and rebuilders for :mod:`grpo.pipeline`.

This module encapsulates reading existing evaluation artifacts from disk and
reconstructing metrics from predictions when partial runs are detected.
"""

from __future__ import annotations

import json
import logging
import importlib as _importlib  # avoid triggering gpt4o.core lazy attribute sentinel
import math
from pathlib import Path
from typing import Iterable, Mapping

from common.opinion import DEFAULT_SPECS
from common.opinion.metrics import compute_opinion_metrics
from common.pipeline.io import write_metrics_json, iter_jsonl_rows
from common.opinion.baselines import baseline_metrics as opinion_baseline_metrics
from common.evaluation import slate_eval
from common.evaluation.prediction_rows import (
    observation_from_row as _obs_from_row_common,
)
from .next_video import NextVideoEvaluationResult
from .opinion import (
    OpinionEvaluationResult,
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
)

gpt4o_utils = _importlib.import_module("gpt4o.core.utils")


LOGGER = logging.getLogger("grpo.pipeline")


def _load_json(path: Path) -> Mapping[str, object]:
    """Load JSON payload from ``path``."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl_rows(path: Path) -> Iterable[Mapping[str, object]]:
    """Compatibility wrapper around :func:`common.pipeline.io.iter_jsonl_rows`."""

    yield from iter_jsonl_rows(path, ignore_errors=True)


def _observation_from_row(row: Mapping[str, object]) -> slate_eval.Observation:
    """Build a slate-eval Observation from a predictions row."""

    raw_output = str(row.get("gpt_output") or "")
    is_formatted = bool(gpt4o_utils.ANS_TAG.search(raw_output)) if raw_output else False
    return _obs_from_row_common(row, is_formatted=is_formatted)


def _collect_opinion_vectors(
    predictions_path: Path,
) -> tuple[list[float], list[float], list[float]]:
    """Collect truth-before, truth-after and predicted-after vectors.

    Skips rows with malformed numbers or NaNs.
    """

    truth_before: list[float] = []
    truth_after: list[float] = []
    pred_after: list[float] = []
    for row in _iter_jsonl_rows(predictions_path):
        try:
            before_val = float(row.get("before"))
            after_val = float(row.get("after"))
            pred_val = float(row.get("prediction"))
        except (TypeError, ValueError):
            continue
        if any(math.isnan(v) for v in (before_val, after_val, pred_val)):
            continue
        # Normalise predictions to [0, 1] when they appear in [1, 7]
        if pred_val > 1.0:
            pred_val = (pred_val - 1.0) / 6.0
        truth_before.append(before_val)
        truth_after.append(after_val)
        pred_after.append(pred_val)
    return truth_before, truth_after, pred_after


def _load_next_video_from_disk(run_dir: Path) -> NextVideoEvaluationResult | None:
    """Return a :class:`NextVideoEvaluationResult` by reading existing metrics.

    Rebuilds metrics from predictions.jsonl when necessary (partial runs).
    """

    def _resolve_effective_run_dir(candidate: Path) -> Path:
        """Return a run directory that actually contains outputs.

        Supports legacy layouts where outputs were written under
        ``<run_dir>/<label>`` by selecting a single immediate child that has
        metrics/predictions when the top-level is empty. If multiple children
        exist, prefer the one with the newest metrics.json or predictions.
        """

        top_metrics = candidate / "metrics.json"
        top_preds = candidate / "predictions.jsonl"
        if top_metrics.exists() or top_preds.exists():
            return candidate

        # Look one level down for a viable child directory
        children = [p for p in candidate.glob("*") if p.is_dir()]
        viable: list[tuple[float, Path]] = []
        for child in children:
            child_metrics_path = child / "metrics.json"
            child_predictions_path = child / "predictions.jsonl"
            if child_metrics_path.exists() or child_predictions_path.exists():
                # Use latest mtime of either artifact as the ranking key
                try:
                    metrics_mtime = (
                        child_metrics_path.stat().st_mtime
                        if child_metrics_path.exists()
                        else 0.0
                    )
                    predictions_mtime = (
                        child_predictions_path.stat().st_mtime
                        if child_predictions_path.exists()
                        else 0.0
                    )
                    latest_mtime = max(metrics_mtime, predictions_mtime)
                except OSError:
                    latest_mtime = 0.0
                viable.append((latest_mtime, child))
        if not viable:
            return candidate
        # Pick the newest viable child
        viable.sort(key=lambda t: t[0], reverse=True)
        return viable[0][1]

    run_dir = _resolve_effective_run_dir(run_dir)

    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"
    qa_log_path = run_dir / "qa.log"

    # Fast-path when metrics exist.
    if metrics_path.exists():
        metrics = _load_json(metrics_path)
        return NextVideoEvaluationResult(
            run_dir=run_dir,
            metrics_path=metrics_path,
            predictions_path=predictions_path,
            qa_log_path=qa_log_path,
            metrics=metrics,
        )

    # Fallback: rebuild metrics from predictions.jsonl for partial runs.
    if not predictions_path.exists():
        LOGGER.warning(
            "Next-video predictions not found at %s; cannot rebuild partial metrics.",
            predictions_path,
        )
        return None

    LOGGER.info(
        "Rebuilding next-video metrics from predictions at %s (partial run detected).",
        predictions_path,
    )
    accumulator = slate_eval.EvaluationAccumulator()
    total_rows = 0
    for row in _iter_jsonl_rows(predictions_path):
        total_rows += 1
        accumulator.observe(_observation_from_row(row))

    # Build a minimal request – dataset/split/filters may be unknown here.
    request = slate_eval.SlateMetricsRequest(
        model_name=run_dir.name,
        dataset_name="cached",
        eval_split="unknown",
        filters=slate_eval.EvaluationFilters(issues=[], studies=[]),
    )
    metrics = accumulator.metrics_payload(request)
    try:
        write_metrics_json(metrics_path, metrics)
    except OSError:  # pragma: no cover - defensive
        LOGGER.warning(
            "Unable to write rebuilt next-video metrics to %s", metrics_path
        )

    LOGGER.info(
        "Rebuilt next-video metrics over %d predictions: "
        "accuracy=%.3f parsed=%.3f formatted=%.3f",
        total_rows,
        accumulator.accuracy(),
        accumulator.parsed_rate(),
        accumulator.format_rate(),
    )
    return NextVideoEvaluationResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        qa_log_path=qa_log_path,
        metrics=metrics,
    )


def _resolve_opinion_spec(key: str):
    """Return the opinion spec matching ``key`` (case-insensitive)."""

    lowered = key.lower()
    for spec in DEFAULT_SPECS:
        if spec.key in {key, lowered}:
            return spec
    return None


def _build_opinion_study(study_dir: Path) -> OpinionStudyResult | None:
    """Return an :class:`OpinionStudyResult` derived from disk caches."""

    if not study_dir.is_dir():
        return None
    metrics_path = study_dir / "metrics.json"
    predictions_path = study_dir / "predictions.jsonl"
    # Rebuild per-study metrics from predictions when missing.
    if not metrics_path.exists() and predictions_path.exists():
        LOGGER.info(
            "Recomputing opinion study metrics from %s (partial run).",
            predictions_path,
        )
        truth_before, truth_after, pred_after = _collect_opinion_vectors(
            predictions_path
        )
        metrics = compute_opinion_metrics(
            truth_after=truth_after,
            truth_before=truth_before,
            pred_after=pred_after,
            direction_tolerance=1e-6,
        )
        # Participants reflect number of prediction rows; store inside metrics for loaders.
        metrics = dict(metrics)
        metrics["participants"] = len(truth_after)
        # Use shared baseline helper (no-change + global mean stats).
        baseline = opinion_baseline_metrics(truth_before, truth_after)
        try:
            write_metrics_json(metrics_path, {"metrics": metrics, "baseline": baseline})
        except OSError:  # pragma: no cover - defensive
            LOGGER.warning(
                "Unable to write rebuilt opinion metrics to %s", metrics_path
            )
    if not metrics_path.exists():
        return None
    payload = _load_json(metrics_path)
    metrics = payload.get("metrics", payload)
    # Heuristic: if MAE is clearly on a 1–7 scale (>1.0) or predictions include
    # values outside [0, 1], rebuild metrics from predictions with normalisation.
    need_rescale = False
    try:
        mae_val = float(metrics.get("mae_after", float("nan")))
        if math.isfinite(mae_val) and mae_val > 1.0:
            need_rescale = True
    except (TypeError, ValueError):
        pass
    if predictions_path.exists() and not need_rescale:
        # Peek a few rows for >1 predictions
        try:
            with predictions_path.open("r", encoding="utf-8") as handle:
                for _ in range(10):
                    line = handle.readline()
                    if not line:
                        break
                    try:
                        row = json.loads(line)
                        pred_val = float(row.get("prediction"))
                        if pred_val > 1.0:
                            need_rescale = True
                            break
                    except Exception:
                        continue
        except OSError:
            pass
    if need_rescale and predictions_path.exists():
        LOGGER.info(
            "Rescaling opinion study metrics from %s due to scale mismatch.",
            predictions_path,
        )
        truth_before, truth_after, pred_after = _collect_opinion_vectors(
            predictions_path
        )
        metrics = compute_opinion_metrics(
            truth_after=truth_after,
            truth_before=truth_before,
            pred_after=pred_after,
            direction_tolerance=1e-6,
        )
        metrics = dict(metrics)
        metrics["participants"] = metrics.get("eligible", 0)
        baseline = opinion_baseline_metrics(truth_before, truth_after)
        try:
            write_metrics_json(metrics_path, {"metrics": metrics, "baseline": baseline})
            payload = {"metrics": metrics, "baseline": baseline}
        except OSError:
            LOGGER.warning("Unable to overwrite rescaled opinion metrics at %s", metrics_path)
    spec = _resolve_opinion_spec(study_dir.name)
    if spec is None:
        LOGGER.warning("Unknown opinion study directory %s; skipping.", study_dir)
        return None
    files = OpinionStudyFiles(
        metrics=metrics_path,
        predictions=predictions_path,
        qa_log=study_dir / "qa.log",
    )
    summary = OpinionStudySummary(
        metrics=metrics,
        baseline=payload.get("baseline", {}),
        participants=int(
            metrics.get("participants")
            or sum(1 for _ in _iter_jsonl_rows(predictions_path))
        ),
        eligible=int(metrics.get("eligible", 0)),
    )
    return OpinionStudyResult(
        study=spec,
        files=files,
        summary=summary,
    )


def _load_opinion_from_disk(out_dir: Path) -> OpinionEvaluationResult | None:
    """Return an :class:`OpinionEvaluationResult` using cached metrics.

    Recomputes combined metrics when missing by aggregating prediction vectors
    across available studies.
    """

    combined_path = out_dir / "combined_metrics.json"

    # Always attempt to load per-study results (with fallback rebuilds).
    studies: list[OpinionStudyResult] = []
    for study_dir in sorted(out_dir.glob("*")):
        study = _build_opinion_study(study_dir)
        if study is not None:
            studies.append(study)

    combined_metrics: Mapping[str, object] = {}
    if combined_path.exists():
        combined_payload = _load_json(combined_path)
        combined_metrics = combined_payload.get("metrics", combined_payload)
    else:
        # Recompute combined metrics from available prediction vectors.
        truth_before_all: list[float] = []
        truth_after_all: list[float] = []
        pred_after_all: list[float] = []
        for study_dir in sorted(out_dir.glob("*")):
            predictions_path = study_dir / "predictions.jsonl"
            if not predictions_path.exists():
                continue
            for row in _iter_jsonl_rows(predictions_path):
                try:
                    before_val = float(row.get("before"))
                    after_val = float(row.get("after"))
                    pred_val = float(row.get("prediction"))
                except (TypeError, ValueError):
                    continue
                if any(math.isnan(v) for v in (before_val, after_val, pred_val)):
                    continue
                truth_before_all.append(before_val)
                truth_after_all.append(after_val)
                pred_after_all.append(pred_val)
        if truth_after_all and len(truth_after_all) == len(pred_after_all):
            combined_metrics = compute_opinion_metrics(
                truth_after=truth_after_all,
                truth_before=truth_before_all[: len(truth_after_all)],
                pred_after=pred_after_all,
                direction_tolerance=1e-6,
            )

    if not combined_metrics:
        LOGGER.warning(
            "Opinion combined metrics missing; run evaluation or provide combined_metrics.json",
        )
        return (
            None
            if not studies
            else OpinionEvaluationResult(studies=studies, combined_metrics={})
        )

    return OpinionEvaluationResult(studies=studies, combined_metrics=combined_metrics)

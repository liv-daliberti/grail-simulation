#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for rebuilding GPT-4o reports from cached artefacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

from common.pipeline.io import load_metrics_json
from common.opinion.metrics import compute_opinion_metrics

from .pipeline_models import PipelinePaths, SweepOutcome, coerce_float, parse_config_label
from .pipeline_reports import generate_reports
from .opinion import (
    OpinionArtifacts,
    OpinionEvaluationResult,
    OpinionMetricBundle,
    OpinionStudyResult,
)
from .utils import qa_log_path_for

LOGGER = logging.getLogger("gpt4o.pipeline.cache")


def _load_sweep_outcomes_from_disk(sweep_dir: Path) -> List[SweepOutcome]:
    """
    Return sweep outcomes reconstructed from existing artefacts.

    :param sweep_dir: Directory containing per-configuration sweep outputs.
    :returns: List of :class:`SweepOutcome` instances discovered on disk.
    """
    outcomes: List[SweepOutcome] = []
    if not sweep_dir.exists():
        return outcomes
    for candidate in sorted(sweep_dir.iterdir()):
        if not candidate.is_dir():
            continue
        metrics_path = candidate / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = load_metrics_json(metrics_path)
        except (FileNotFoundError, json.JSONDecodeError):
            LOGGER.warning(
                "[REPORTS] Skipping sweep artefacts at %s (unreadable metrics).",
                metrics_path,
            )
            continue
        try:
            config = parse_config_label(candidate.name)
        except ValueError:
            LOGGER.warning(
                "[REPORTS] Skipping sweep directory '%s' (invalid label).",
                candidate.name,
            )
            continue
        outcomes.append(
            SweepOutcome(
                config=config,
                accuracy=coerce_float(metrics.get("accuracy_overall")),
                parsed_rate=coerce_float(metrics.get("parsed_rate")),
                format_rate=coerce_float(metrics.get("format_rate")),
                metrics_path=metrics_path,
                metrics=metrics,
            )
        )
    outcomes.sort(key=lambda item: item.config.label())
    return outcomes


def _load_selected_outcome_from_disk(
    paths: PipelinePaths, outcomes: List[SweepOutcome]
) -> Tuple[SweepOutcome, Mapping[str, object]]:
    """
    Resolve the selected outcome and final metrics from cached artefacts.

    :param paths: Aggregated pipeline paths for locating artefacts.
    :param outcomes: Sweep outcomes previously reconstructed from disk.
    :returns: Tuple of selected outcome and its metrics mapping.
    :raises RuntimeError: If no sweep outcomes are available.
    """
    selected: SweepOutcome | None = None
    final_metrics: Mapping[str, object] | None = None

    if paths.final_out_dir.exists():
        for directory in sorted(paths.final_out_dir.iterdir()):
            if not directory.is_dir():
                continue
            metrics_path = directory / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                metrics = load_metrics_json(metrics_path)
            except (FileNotFoundError, json.JSONDecodeError):
                LOGGER.warning(
                    "[REPORTS] Skipping final artefacts at %s (unreadable metrics).",
                    metrics_path,
                )
                continue
            label = directory.name
            match = next(
                (outcome for outcome in outcomes if outcome.config.label() == label),
                None,
            )
            if match is None:
                try:
                    config = parse_config_label(label)
                except ValueError:
                    LOGGER.warning(
                        "[REPORTS] Skipping final directory '%s' (invalid label).",
                        label,
                    )
                    continue
                match = SweepOutcome(
                    config=config,
                    accuracy=coerce_float(metrics.get("accuracy_overall")),
                    parsed_rate=coerce_float(metrics.get("parsed_rate")),
                    format_rate=coerce_float(metrics.get("format_rate")),
                    metrics_path=metrics_path,
                    metrics=metrics,
                )
                outcomes.append(match)
                outcomes.sort(key=lambda item: item.config.label())
            selected = match
            final_metrics = metrics
            break

    if selected is None:
        if not outcomes:
            raise RuntimeError(
                "No GPT-4o sweep artefacts were found; run sweeps before generating reports."
            )
        selected = max(
            outcomes,
            key=lambda outcome: (outcome.accuracy, outcome.parsed_rate, outcome.format_rate),
        )
        final_metrics = selected.metrics

    assert final_metrics is not None  # narrow Optional for type checkers
    return selected, final_metrics


def _load_prediction_vectors(
    predictions_path: Path,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Return opinion prediction vectors parsed from ``predictions_path``.

    :param predictions_path: Path to the JSONL predictions file.
    :returns: Tuple of ``(truth_before, truth_after, pred_after)`` vectors.
    """
    truth_before: List[float] = []
    truth_after: List[float] = []
    pred_after: List[float] = []
    if not predictions_path.exists():
        return truth_before, truth_after, pred_after
    try:
        with predictions_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    before = float(row.get("before"))
                    after = float(row.get("after"))
                    predicted = float(row.get("predicted_after"))
                except (TypeError, ValueError):
                    continue
                truth_before.append(before)
                truth_after.append(after)
                pred_after.append(predicted)
    except OSError:
        LOGGER.warning(
            "[REPORTS] Unable to read opinion predictions at %s.", predictions_path
        )
    return truth_before, truth_after, pred_after


def _load_opinion_result_from_disk(
    paths: PipelinePaths, config_label: str
) -> OpinionEvaluationResult | None:
    """
    Rehydrate opinion evaluation results from cached artefacts.

    :param paths: Aggregated pipeline paths pointing to opinion outputs.
    :param config_label: Configuration label identifying the promoted run.
    :returns: Fully populated :class:`OpinionEvaluationResult` or ``None`` if nothing was found.
    """
    opinion_root = paths.opinion_dir / config_label
    if not opinion_root.exists():
        return None

    combined_before: List[float] = []
    combined_after: List[float] = []
    combined_pred: List[float] = []
    studies: Dict[str, OpinionStudyResult] = {}

    for study_dir in sorted(opinion_root.iterdir()):
        if not study_dir.is_dir():
            continue
        metrics_path = study_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning(
                "[REPORTS] Skipping opinion study at %s (unreadable metrics).",
                metrics_path,
            )
            continue

        metrics = payload.get("metrics", {}) or {}
        baseline = payload.get("baseline", {}) or {}
        participants = int(payload.get("participants", 0) or 0)
        study_key = str(payload.get("study", study_dir.name))
        study_label = str(payload.get("study_label", study_key))
        issue = str(payload.get("issue", ""))
        eligible = int(metrics.get("eligible", participants))

        predictions_path = study_dir / "predictions.jsonl"
        before_vals, after_vals, pred_vals = _load_prediction_vectors(predictions_path)
        if before_vals:
            combined_before.extend(before_vals)
        if after_vals:
            combined_after.extend(after_vals)
        if pred_vals:
            combined_pred.extend(pred_vals)

        if not participants and after_vals:
            participants = len(after_vals)
        if not eligible and after_vals:
            eligible = len(after_vals)

        artifacts = OpinionArtifacts(
            metrics=metrics_path,
            predictions=predictions_path,
            qa_log=qa_log_path_for(study_dir),
        )
        bundle = OpinionMetricBundle(metrics=metrics, baseline=baseline)
        studies[study_key] = OpinionStudyResult(
            study_key=study_key,
            study_label=study_label,
            issue=issue,
            participants=participants,
            eligible=eligible,
            artifacts=artifacts,
            metric_bundle=bundle,
        )

    if not studies:
        return None

    combined_metrics: Mapping[str, object] = {}
    if combined_after and len(combined_after) == len(combined_pred):
        combined_metrics = compute_opinion_metrics(
            truth_after=combined_after,
            truth_before=combined_before[: len(combined_after)],
            pred_after=combined_pred,
            direction_tolerance=1e-6,
        )

    return OpinionEvaluationResult(
        studies=studies,
        combined_metrics=combined_metrics,
        config_label=config_label,
    )


def run_reports_stage(paths: PipelinePaths, *, repo_root: Path) -> None:
    """
    Generate GPT-4o reports using cached sweep and final artefacts.

    :param paths: Aggregated pipeline paths used to locate artefacts.
    :param repo_root: Repository root for relative path conversion.
    :returns: ``None``.
    :raises RuntimeError: If required sweep artefacts are missing.
    """
    LOGGER.info("Rebuilding GPT-4o reports from existing artefacts.")
    outcomes = _load_sweep_outcomes_from_disk(paths.sweep_dir)
    if not outcomes:
        raise RuntimeError(
            f"No GPT-4o sweep metrics found under {paths.sweep_dir}. "
            "Run the pipeline sweeps before rebuilding reports."
        )
    selected, final_metrics = _load_selected_outcome_from_disk(paths, outcomes)
    opinion_result = _load_opinion_result_from_disk(paths, selected.config.label())

    generate_reports(
        reports_dir=paths.reports_dir,
        outcomes=outcomes,
        selected=selected,
        final_metrics=final_metrics,
        opinion_result=opinion_result,
        repo_root=repo_root,
    )
    LOGGER.info("GPT-4o reports refreshed at %s.", paths.reports_dir)


__all__ = ["run_reports_stage"]

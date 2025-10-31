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

"""Helper utilities for GPT-4o opinion evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple, cast

from importlib import import_module
import numpy as np
_common_opinion = import_module("common.opinion")
compute_opinion_metrics = _common_opinion.compute_opinion_metrics
float_or_none = _common_opinion.float_or_none


def document_from_example(example: Mapping[str, object]) -> str:
    """Assemble the viewer profile/state text bundle used for opinion prompts.

    :param example: Row from the opinion dataset containing viewer context
        fields such as ``viewer_profile`` and ``state_text`` plus optional
        ``current_video_title``/``next_video_title`` hints.
    :returns: Multi-section string that includes a "Viewer profile", optional
        "Context" block, and "Currently watching"/"Next video shown" lines when
        available. Empty sections are omitted and surrounding whitespace is
        trimmed.
    :rtype: str
    """
    profile_source = (
        example.get("viewer_profile")
        or example.get("viewer_profile_sentence")
        or ""
    )
    profile = str(profile_source).strip()
    state_text = str(example.get("state_text") or "").strip()

    sections: List[str] = []
    if profile:
        sections.append(f"Viewer profile:\n{profile}")
    if state_text:
        sections.append(f"Context:\n{state_text}")

    current_title = str(example.get("current_video_title") or "").strip()
    next_title = str(example.get("next_video_title") or "").strip()
    if current_title:
        sections.append(f"Currently watching: {current_title}")
    if next_title:
        sections.append(f"Next video shown: {next_title}")

    return "\n\n".join(sections).strip()


@dataclass(frozen=True)
class CollectExampleHooks:
    """Dependency injection hooks used while collecting opinion examples.

    These hooks allow tests or callers to override parsing/formatting behaviour
    without modifying the core logic.

    - ``float_parser`` converts raw values into ``float`` or ``None``.
    - ``document_builder`` assembles the viewer context string.
    """

    float_parser: Callable[[object], float | None] = float_or_none
    document_builder: Callable[[Mapping[str, object]], str] = document_from_example


@dataclass(frozen=True)
class CollectExamplesConfig:
    """Configuration controlling opinion example collection and filtering.

    :param allows: Predicate receiving ``(issue, study)`` and returning whether
        the example should be included.
    :param eval_max: Optional cap on retained participants (0 keeps all).
    :param hooks: Pluggable hooks for parsing floats and building documents.
    """

    allows: Callable[[str, str], bool]
    eval_max: int
    hooks: CollectExampleHooks = field(default_factory=CollectExampleHooks)


def clip_prediction(value: float) -> float:
    """Clamp predictions to the 1–7 opinion index range.

    :param value: Raw predicted post-study opinion index.
    :returns: ``value`` clipped to ``[1.0, 7.0]``.
    :rtype: float
    """
    return max(1.0, min(7.0, float(value)))


def baseline_metrics(
    truth_before: Sequence[float], truth_after: Sequence[float]
) -> Dict[str, object]:
    """Compute baseline metrics mirroring the KNN/XGB implementations.

    The baselines cover two simple predictors:
    - Global mean of ``truth_after`` for MAE/RMSE.
    - No-change model using ``truth_before`` as the prediction for direction and
      calibration baselines.

    :param truth_before: Ground-truth pre-study opinion indices per participant.
    :param truth_after: Ground-truth post-study opinion indices per participant.
    :returns: Mapping with keys such as ``global_mean_after``, ``mae_global_mean_after``,
        ``rmse_global_mean_after``, and derivatives from the no-change baseline
        (``mae_using_before``, ``rmse_using_before``, ``direction_accuracy``,
        ``calibration_*``, ``kl_divergence_change_zero``).
    :rtype: dict[str, object]
    """
    after_arr = np.asarray(truth_after, dtype=np.float32)
    before_arr = np.asarray(truth_before, dtype=np.float32)
    if after_arr.size == 0:
        return {}

    baseline_mean = float(np.mean(after_arr))
    baseline_predictions = np.full_like(after_arr, baseline_mean)
    mae_mean = float(np.mean(np.abs(baseline_predictions - after_arr)))
    rmse_mean = float(np.sqrt(np.mean((baseline_predictions - after_arr) ** 2)))

    no_change = compute_opinion_metrics(
        truth_after=after_arr,
        truth_before=before_arr,
        pred_after=before_arr,
    )
    direction_accuracy = no_change.get("direction_accuracy")
    direction_accuracy = (
        float(direction_accuracy) if isinstance(direction_accuracy, (int, float)) else None
    )

    return {
        "global_mean_after": baseline_mean,
        "mae_global_mean_after": mae_mean,
        "rmse_global_mean_after": rmse_mean,
        "mae_using_before": float(no_change.get("mae_after", float("nan"))),
        "rmse_using_before": float(no_change.get("rmse_after", float("nan"))),
        "mae_change_zero": float(no_change.get("mae_change", float("nan"))),
        "rmse_change_zero": float(no_change.get("rmse_change", float("nan"))),
        "calibration_slope_change_zero": no_change.get("calibration_slope"),
        "calibration_intercept_change_zero": no_change.get("calibration_intercept"),
        "calibration_ece_change_zero": no_change.get("calibration_ece"),
        "calibration_bins_change_zero": no_change.get("calibration_bins"),
        "kl_divergence_change_zero": no_change.get("kl_divergence_change"),
        "direction_accuracy": direction_accuracy,
    }


def load_materialised_split(
    dataset: object, preferred_split: str
) -> Tuple[str, Iterable[Mapping[str, object]]]:
    """Return the evaluation split from either a DatasetDict or single dataset.

    :param dataset: Hugging Face ``DatasetDict`` or a single split-like dataset.
    :param preferred_split: Preferred split name (e.g. ``"validation"``).
    :returns: Tuple of ``(resolved_split_name, iterable_rows)``. Falls back to a
        sensible existing split when the preferred one is missing.
    :rtype: tuple[str, Iterable[Mapping[str, object]]]
    :raises RuntimeError: If ``dataset`` is ``None``.
    """
    if dataset is None:
        raise RuntimeError("Expected dataset to be materialised before opinion evaluation.")

    eval_split = preferred_split
    available: List[str] = []
    if hasattr(dataset, "keys"):
        try:
            available = list(cast(Mapping[str, object], dataset).keys())
        except (TypeError, AttributeError):
            available = []
    if available:
        for candidate in (preferred_split, "validation", "eval", "test", "train"):
            if candidate in available:
                eval_split = candidate
                break
        else:
            eval_split = available[0]
        split_dataset = cast(Mapping[str, Iterable[Mapping[str, object]]], dataset)[
            eval_split
        ]
    else:
        split_dataset = dataset
    return eval_split, split_dataset


def _normalise_example_entry(
    entry: Mapping[str, object],
    *,
    spec,
    issue: str,
    study: str,
    hooks: CollectExampleHooks,
) -> Mapping[str, object] | None:
    """Return a participant payload when ``entry`` satisfies the study spec.

    :param entry: Raw dataset row for an individual participant/time step.
    :param spec: :class:`~common.opinion.OpinionSpec` with column definitions.
    :param issue: Issue label associated with the row.
    :param study: Participant study key associated with the row.
    :param hooks: Parsing/formatting hooks used to coerce inputs.
    :returns: Canonical participant payload or ``None`` if the row is
        incomplete (missing ids, numeric fields, or empty document).
    :rtype: dict[str, object] | None
    """

    before = hooks.float_parser(entry.get(spec.before_column))
    after = hooks.float_parser(entry.get(spec.after_column))
    if before is None or after is None:
        return None
    participant_id = str(entry.get("participant_id") or "").strip()
    if not participant_id:
        return None
    document = hooks.document_builder(entry)
    if not document:
        return None
    try:
        step_index = int(entry.get("step_index") or -1)
    except (TypeError, ValueError):
        step_index = -1
    return {
        "participant_id": participant_id,
        "document": document,
        "before": before,
        "after": after,
        "issue": issue,
        "study": study,
        "step_index": step_index,
        "raw": entry,
    }


def collect_examples(
    rows: Sequence[Mapping[str, object]],
    spec,
    config: CollectExamplesConfig,
) -> Tuple[List[Mapping[str, object]], int]:
    """Return filtered participant examples for the provided study spec.

    Deduplicates by ``participant_id`` keeping the last time step, applies
    filters, and optionally caps the number of retained participants.

    :param rows: Materialised dataset rows for the evaluation split.
    :param spec: :class:`~common.opinion.OpinionSpec` describing column names.
    :param config: Collection and filtering options including hooks.
    :returns: Tuple of ``(retained_examples, original_count)`` where
        ``original_count`` reflects the number of unique participants prior to
        ``eval_max`` truncation.
    :rtype: tuple[list[Mapping[str, object]], int]
    """
    per_participant: MutableMapping[str, Tuple[int, Mapping[str, object]]] = {}

    for entry in rows:
        issue = str(entry.get("issue") or "").strip()
        study = str(entry.get("participant_study") or "").strip()
        if not config.allows(issue, study):
            continue
        if study.lower() != spec.key.lower():
            continue
        payload = _normalise_example_entry(
            entry,
            spec=spec,
            issue=issue,
            study=study,
            hooks=config.hooks,
        )
        if payload is None:
            continue
        participant_id = payload["participant_id"]
        existing = per_participant.get(participant_id)
        step_index = payload["step_index"]
        if existing is None or step_index >= existing[0]:
            per_participant[participant_id] = (step_index, payload)

    retained = sorted(
        (payload for _, payload in per_participant.values()),
        key=lambda item: (item["participant_id"], item["step_index"]),
    )
    original_count = len(retained)
    eval_max = config.eval_max
    if eval_max and len(retained) > eval_max:
        retained = retained[: eval_max]
    return retained, original_count


__all__ = [
    "baseline_metrics",
    "clip_prediction",
    "collect_examples",
    "CollectExampleHooks",
    "CollectExamplesConfig",
    "document_from_example",
    "float_or_none",
    "load_materialised_split",
]

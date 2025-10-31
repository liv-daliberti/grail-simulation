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

"""Opinion-shift evaluation for finetuned GRPO checkpoints."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Mapping, MutableMapping, Sequence

import numpy as np

from common.opinion import (
    DEFAULT_SPECS,
    OpinionArtifacts,
    OpinionStudyResult as _BaseOpinionStudyResult,
    OpinionSpec,
    compute_opinion_metrics,
    format_opinion_user_prompt,
)
from common.pipeline.io import write_metrics_json, write_segmented_markdown_log
from common.open_r1.example_utils import row_to_training_example

from .dataset import load_dataset_split
from .model import generate_chat_completion

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import GenerationSettings, ModelLike, TokenizerLike

LOGGER = logging.getLogger("grpo.opinion")

ANSWER_PATTERN = re.compile(r"(?si)<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>")
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")

def _clip_prediction(value: float) -> float:
    """Clamp predictions to the 1–7 opinion scale.

    :param value: Raw numeric opinion prediction.
    :returns: Value clipped to the inclusive range ``[1.0, 7.0]``.
    :rtype: float
    """

    return min(7.0, max(1.0, value))


def _compute_baseline_metrics(
    *, truth_before: Sequence[float], truth_after: Sequence[float]
) -> Mapping[str, object]:
    """Return baseline opinion metrics matching the GPT-4o implementation."""

    after_arr = np.asarray(truth_after, dtype=np.float32)
    before_arr = np.asarray(truth_before, dtype=np.float32)
    if after_arr.size == 0:
        return {}

    mean_after = float(after_arr.mean())
    mean_predictions = np.full_like(after_arr, mean_after)
    mae_mean = float(np.mean(np.abs(mean_predictions - after_arr)))
    rmse_mean = float(np.sqrt(np.mean((mean_predictions - after_arr) ** 2)))

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
        "global_mean_after": mean_after,
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


def _parse_prediction(raw_output: str) -> float:
    """Parse a numeric prediction from the model output.

    :param raw_output: Model completion containing the opinion estimate.
    :returns: Parsed float prediction or ``NaN`` when parsing fails.
    :rtype: float
    """

    match = ANSWER_PATTERN.search(raw_output)
    candidate = match.group(1) if match else None
    if not candidate:
        fallback = NUMBER_PATTERN.search(raw_output)
        candidate = fallback.group(0) if fallback else ""
    try:
        return _clip_prediction(float(candidate))
    except (TypeError, ValueError):
        LOGGER.warning("Unable to parse opinion prediction from output: %r", raw_output)
        return float("nan")


def _extract_user_prompt(messages: Sequence[Mapping[str, str]]) -> str:
    """Return the most recent user message from ``messages``.

    :param messages: Chat transcript previously supplied to the model.
    :returns: Latest user message or an empty string when absent.
    :rtype: str
    """

    for message in reversed(messages):
        if message.get("role") == "user" and message.get("content"):
            return str(message["content"]).strip()
    return ""


def _select_examples_by_participant(
    rows: Sequence[Mapping[str, object]],
    *,
    spec: OpinionSpec,
) -> List[Mapping[str, object]]:
    """Return the latest example per participant for ``spec``.

    :param rows: Materialised dataset rows.
    :param spec: Opinion study specification driving selection.
    :returns: Ordered list containing at most one example per participant.
    :rtype: list[Mapping[str, object]]
    """

    per_participant: MutableMapping[str, tuple[int, Mapping[str, object]]] = {}
    for row in rows:
        issue = str(row.get("issue") or "").strip().lower()
        study = str(row.get("participant_study") or "").strip().lower()
        if issue != spec.issue.lower() or study != spec.key.lower():
            continue
        participant_id = str(row.get("participant_id") or "").strip()
        if not participant_id:
            continue
        before = row.get(spec.before_column)
        after = row.get(spec.after_column)
        if before is None or after is None:
            continue
        try:
            before_value = float(before)
            after_value = float(after)
        except (TypeError, ValueError):
            continue
        try:
            step_index = int(row.get("step_index") or -1)
        except (TypeError, ValueError):
            step_index = -1
        payload = dict(row)
        payload["_participant_id"] = participant_id
        payload["_before"] = before_value
        payload["_after"] = after_value
        payload["_step_index"] = step_index
        existing = per_participant.get(participant_id)
        if existing is None or step_index >= existing[0]:
            per_participant[participant_id] = (step_index, payload)
    selected = [payload for _, payload in per_participant.values()]
    selected.sort(key=lambda item: (item["_participant_id"], item["_step_index"]))
    return selected


@dataclass(frozen=True)
class OpinionDatasetSpec:
    """Dataset selection applied for opinion evaluation.

    :ivar str name: Local path or Hugging Face dataset identifier.
    :ivar str split: Dataset split used when loading rows.
    :ivar str | None cache_dir: Optional datasets cache directory.
    """

    name: str
    split: str
    cache_dir: str | None


@dataclass(frozen=True)
class OpinionPromptSettings:
    """Prompt configuration used when preparing model inputs.

    :ivar str system: Baseline system prompt passed to the model.
    :ivar str opinion: Opinion-specific system prompt for regression tasks.
    :ivar str | None solution_key: Optional dataset column containing gold ids.
    :ivar int max_history: Maximum number of conversation turns to forward.
    """

    system: str
    opinion: str
    solution_key: str | None
    max_history: int


@dataclass(frozen=True)
class OpinionEvaluationControls:
    """Execution controls covering participant limits and caching.

    :ivar int max_participants: Upper bound on participants per study (0 => unlimited).
    :ivar float direction_tolerance: Tolerance used for direction comparison.
    :ivar bool overwrite: Whether to overwrite existing opinion artefacts.
    """

    max_participants: int
    direction_tolerance: float
    overwrite: bool


@dataclass(frozen=True)
class OpinionEvaluationSettings:
    """Configuration for GRPO opinion evaluation.

    :ivar OpinionDatasetSpec dataset: Dataset selection parameters.
    :ivar OpinionPromptSettings prompts: Prompt configuration bundle.
    :ivar OpinionEvaluationControls controls: Participant-level execution controls.
    :ivar Sequence[str] | None include_studies: Optional whitelist of study keys.
    """

    dataset: OpinionDatasetSpec
    prompts: OpinionPromptSettings
    controls: OpinionEvaluationControls
    include_studies: Sequence[str] | None


OpinionStudyFiles = OpinionArtifacts


@dataclass(frozen=True)
class OpinionStudySummary:
    """Bundle capturing evaluation metrics and participant counts for a study."""

    metrics: Mapping[str, object]
    baseline: Mapping[str, object]
    participants: int
    eligible: int


class OpinionStudyResult(_BaseOpinionStudyResult):
    """Per-study artefacts returned by the evaluation runner."""

    def __init__(
        self,
        *,
        study: OpinionSpec,
        files: OpinionStudyFiles,
        summary: OpinionStudySummary | None = None,
        **legacy_kwargs,
    ) -> None:
        if summary is None:
            try:
                metrics = legacy_kwargs.pop("metrics")
                baseline = legacy_kwargs.pop("baseline")
                participants = legacy_kwargs.pop("participants")
                eligible = legacy_kwargs.pop("eligible")
            except KeyError as exc:  # pragma: no cover - defensive guard
                raise TypeError(
                    "OpinionStudyResult requires either 'summary' or the legacy "
                    "metrics/baseline/participants/eligible arguments."
                ) from exc
            summary = OpinionStudySummary(
                metrics=metrics,
                baseline=baseline,
                participants=int(participants),
                eligible=int(eligible),
            )
        if legacy_kwargs:
            raise TypeError(f"Unexpected arguments for OpinionStudyResult: {sorted(legacy_kwargs)}")
        super().__init__(
            study_key=study.key,
            study_label=study.label,
            issue=study.issue,
            participants=summary.participants,
            eligible=summary.eligible,
            metrics=summary.metrics,
            baseline=summary.baseline,
            artifacts=files,
            spec=study,
        )

    @property
    def study(self) -> OpinionSpec:
        """Return the opinion study specification."""

        spec = self.spec
        if spec is None:  # pragma: no cover - defensive guard
            raise RuntimeError("OpinionStudyResult missing spec reference")
        return spec

    @property
    def files(self) -> OpinionStudyFiles:
        """Expose the filesystem artefacts associated with the study."""

        return self.artifacts


@dataclass(frozen=True)
class OpinionEvaluationResult:
    """Aggregate payload containing per-study and combined metrics."""

    studies: Sequence[OpinionStudyResult]
    combined_metrics: Mapping[str, object]


# Backwards compatibility: retain OpinionArtifacts alias.
OpinionArtifacts = OpinionStudyFiles


@dataclass
class _StudyAccumulator:
    """Capture the rolling state for a single study evaluation.

    :ivar list[Mapping[str, object]] predictions: Serialised participant predictions.
    :ivar list[str] qa_entries: Markdown QA log snippets per participant.
    :ivar list[float] truth_before: Ground-truth pre-study opinion indices.
    :ivar list[float] truth_after: Ground-truth post-study opinion indices.
    :ivar list[float] pred_after: Model-predicted post-study indices.
    """

    predictions: List[Mapping[str, object]] = field(default_factory=list)
    qa_entries: List[str] = field(default_factory=list)
    truth_before: List[float] = field(default_factory=list)
    truth_after: List[float] = field(default_factory=list)
    pred_after: List[float] = field(default_factory=list)

    def record(self, artefact: "_OpinionPredictionArtefact") -> None:
        """Append artefacts produced by a single participant evaluation.

        :param artefact: Prediction artefact captured for one participant.
        :returns: ``None``. Internal buffers are extended in-place.
        """

        self.predictions.append(artefact.payload)
        self.qa_entries.append(artefact.qa_entry)
        self.truth_before.append(artefact.before)
        self.truth_after.append(artefact.after)
        self.pred_after.append(artefact.prediction)

    @property
    def participants(self) -> int:
        """Return the number of participants included in the study.

        :returns: Count of participant predictions stored.
        :rtype: int
        """

        return len(self.predictions)


@dataclass
class _CombinedVectors:
    """Aggregate opinion vectors across every evaluated study.

    :ivar list[float] truth_before: Concatenated pre-study opinion indices.
    :ivar list[float] truth_after: Concatenated post-study opinion indices.
    :ivar list[float] pred_after: Concatenated model predictions.
    """

    truth_before: List[float] = field(default_factory=list)
    truth_after: List[float] = field(default_factory=list)
    pred_after: List[float] = field(default_factory=list)

    def extend(self, accumulator: _StudyAccumulator) -> None:
        """Extend the combined vectors with a study-level accumulator.

        :param accumulator: Study accumulator containing participant vectors.
        :returns: ``None``. Combined state is extended in-place.
        """

        self.truth_before.extend(accumulator.truth_before)
        self.truth_after.extend(accumulator.truth_after)
        self.pred_after.extend(accumulator.pred_after)


@dataclass(frozen=True)
class OpinionInferenceContext:
    """Bundle the runtime inference dependencies used across studies."""

    tokenizer: 'TokenizerLike'
    model: 'ModelLike'
    generation: 'GenerationSettings'


@dataclass(frozen=True)
class OpinionStudyContext:
    """Convenience bundle combining prompts, tolerance, and inference context."""

    prompts: OpinionPromptSettings
    direction_tolerance: float
    inference: OpinionInferenceContext


@dataclass(frozen=True)
class _OpinionPredictionArtefact:
    """Artefacts generated when scoring a single participant example.

    :ivar Mapping[str, object] payload: Serialisable prediction payload.
    :ivar str qa_entry: Markdown QA block for the participant.
    :ivar float before: Pre-study opinion index.
    :ivar float after: Post-study ground-truth opinion index.
    :ivar float prediction: Model-predicted post-study opinion index.
    """

    payload: Mapping[str, object]
    qa_entry: str
    before: float
    after: float
    prediction: float


def _resolve_studies(include: Sequence[str] | None) -> Sequence[OpinionSpec]:
    """Return opinion specs filtered by ``include`` (if provided).

    :param include: Optional whitelist of study keys to evaluate.
    :returns: Sequence of :class:`common.opinion.OpinionSpec` objects.
    :rtype: Sequence[OpinionSpec]
    :raises ValueError: If no studies match the requested keys.
    """

    if not include:
        return DEFAULT_SPECS
    tokens = {token.strip().lower() for token in include if token}
    resolved = [spec for spec in DEFAULT_SPECS if spec.key.lower() in tokens]
    if not resolved:
        raise ValueError(
            f"No opinion studies matched {sorted(tokens)}. "
            f"Expected one of {[spec.key for spec in DEFAULT_SPECS]}."
        )
    return resolved


def _write_predictions(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Persist opinion predictions to JSONL.

    :param path: Destination path for the predictions file.
    :param rows: Iterable of prediction payloads to serialise.
    :returns: ``None``. The file is written to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in rows:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


def _resolve_study_examples(
    rows: Sequence[Mapping[str, object]],
    *,
    spec: OpinionSpec,
    max_participants: int,
) -> Sequence[Mapping[str, object]]:
    """Return at most ``max_participants`` examples for ``spec``."""

    selected = _select_examples_by_participant(rows, spec=spec)
    if max_participants and len(selected) > max_participants:
        LOGGER.info(
            "[OPINION] Limiting study=%s participants to %d (from %d).",
            spec.key,
            max_participants,
            len(selected),
        )
        return selected[:max_participants]
    return selected


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


def _evaluate_study(
    *,
    spec: OpinionSpec,
    examples: Sequence[Mapping[str, object]],
    study_context: OpinionStudyContext,
    files: OpinionStudyFiles,
) -> tuple[OpinionStudyResult, _StudyAccumulator]:
    """Score ``examples`` for ``spec`` and materialise study artefacts."""

    accumulator = _StudyAccumulator()
    for example in examples:
        artefact = _score_opinion_example(
            example=example,
            spec=spec,
            prompts=study_context.prompts,
            context=study_context.inference,
        )
        if artefact is None:
            continue
        accumulator.record(artefact)

    metrics = compute_opinion_metrics(
        truth_after=accumulator.truth_after,
        truth_before=accumulator.truth_before,
        pred_after=accumulator.pred_after,
        direction_tolerance=study_context.direction_tolerance,
    )
    baseline = _compute_baseline_metrics(
        truth_after=accumulator.truth_after,
        truth_before=accumulator.truth_before,
    )

    _persist_study_outputs(
        files,
        metrics=metrics,
        baseline=baseline,
        accumulator=accumulator,
    )

    summary = OpinionStudySummary(
        metrics=metrics,
        baseline=baseline,
        participants=accumulator.participants,
        eligible=int(metrics.get("eligible", 0)),
    )
    result = OpinionStudyResult(
        study=spec,
        files=files,
        summary=summary,
    )
    direction = metrics.get("direction_accuracy")
    mae_after = metrics.get("mae_after")
    LOGGER.info(
        "[OPINION] study=%s issue=%s participants=%d eligible=%d direction=%s mae_after=%s",
        spec.key,
        spec.issue,
        summary.participants,
        summary.eligible,
        f"{float(direction):.3f}" if isinstance(direction, (int, float)) else "nan",
        f"{float(mae_after):.3f}" if isinstance(mae_after, (int, float)) else "nan",
    )
    return result, accumulator


def _score_opinion_example(
    *,
    example: Mapping[str, object],
    spec: OpinionSpec,
    prompts: OpinionPromptSettings,
    context: OpinionInferenceContext,
) -> _OpinionPredictionArtefact | None:
    """Return prediction artefacts for a single participant example.

    :param example: Raw participant row annotated by :func:`_select_examples_by_participant`.
    :param spec: Opinion study specification guiding evaluation.
    :param prompts: Prompt configuration used to build model inputs.
    :param context: Runtime inference context providing model dependencies.
    :returns: Prediction artefact or ``None`` if the example cannot be evaluated.
    :rtype: _OpinionPredictionArtefact | None
    """

    prepared = row_to_training_example(
        example,
        system_prompt=prompts.system,
        solution_key=prompts.solution_key,
        max_history=prompts.max_history,
    )
    if not prepared:
        return None
    messages_raw = prepared.get("prompt")
    if not isinstance(messages_raw, list) or not messages_raw:
        return None
    user_prompt = _extract_user_prompt(messages_raw) or str(
        example.get("state_text") or ""
    ).strip()
    before = float(example["_before"])
    after = float(example["_after"])

    messages = [
        {"role": "system", "content": prompts.opinion},
        {
            "role": "user",
            "content": format_opinion_user_prompt(
                issue_label=spec.issue.replace("_", " ").title(),
                pre_study_index=before,
                viewer_context=user_prompt,
                post_watch_instruction=(
                    "After the participant watches the recommended next video, "
                    "estimate their post-study opinion index."
                ),
            ),
        },
    ]
    raw_output = generate_chat_completion(
        context.model,
        context.tokenizer,
        messages,
        settings=context.generation,
    )
    prediction = _parse_prediction(raw_output)

    return _OpinionPredictionArtefact(
        payload={
            "participant_id": example["_participant_id"],
            "issue": spec.issue,
            "study": spec.key,
            "step_index": int(example["_step_index"]),
            "before": before,
            "after": after,
            "prediction": prediction,
            "raw_output": raw_output,
            "messages": messages,
        },
        qa_entry="\n".join(
            [
                f"## Participant {example['_participant_id']}",
                f"- Study: {spec.label}",
                f"- Before: {before:.2f}",
                f"- After: {after:.2f}",
                "",
                "### Prompt",
                json.dumps(messages, indent=2, ensure_ascii=False),
                "",
                "### Model output",
                raw_output.strip(),
                "",
                f"### Parsed prediction: {prediction:.3f}",
            ]
        ),
        before=before,
        after=after,
        prediction=prediction,
    )


def run_opinion_evaluation(
    *,
    context: OpinionInferenceContext,
    settings: OpinionEvaluationSettings,
    out_dir: Path,
) -> OpinionEvaluationResult:
    """Evaluate opinion regression across configured studies.

    :param context: Runtime inference context shared across studies.
    :param settings: Opinion evaluation configuration bundle.
    :param out_dir: Output directory that will receive study artefacts.
    :returns: Combined opinion evaluation result with per-study metrics.
    :rtype: OpinionEvaluationResult
    """

    LOGGER.info(
        "[OPINION] loading dataset name=%s split=%s include_studies=%s max_participants=%s overwrite=%s",
        settings.dataset.name,
        settings.dataset.split,
        ",".join(settings.include_studies or ()) or "<all>",
        settings.controls.max_participants or "<no-cap>",
        settings.controls.overwrite,
    )
    print(
        "[grpo.opinion] dataset=%s split=%s include_studies=%s max_participants=%s overwrite=%s"
        % (
            settings.dataset.name,
            settings.dataset.split,
            ",".join(settings.include_studies or ()) or "<all>",
            settings.controls.max_participants or "<no-cap>",
            settings.controls.overwrite,
        ),
        flush=True,
    )

    dataset_rows = load_dataset_split(
        settings.dataset.name,
        split=settings.dataset.split,
        cache_dir=settings.dataset.cache_dir,
    )
    LOGGER.info("[OPINION] loaded %d rows for opinion evaluation", len(dataset_rows))
    print(f"[grpo.opinion] loaded {len(dataset_rows)} rows", flush=True)
    combined = _CombinedVectors()
    results: List[OpinionStudyResult] = []
    study_context = OpinionStudyContext(
        prompts=settings.prompts,
        direction_tolerance=settings.controls.direction_tolerance,
        inference=context,
    )

    for spec in _resolve_studies(settings.include_studies):
        examples = _resolve_study_examples(
            dataset_rows,
            spec=spec,
            max_participants=settings.controls.max_participants,
        )
        LOGGER.info(
            "[OPINION] evaluating study=%s participants=%d",
            spec.key,
            len(examples),
        )
        print(
            f"[grpo.opinion] evaluating study={spec.key} participants={len(examples)}",
            flush=True,
        )
        study_result, accumulator = _evaluate_study(
            spec=spec,
            examples=examples,
            study_context=study_context,
            files=_prepare_study_files(out_dir, spec),
        )
        results.append(study_result)
        combined.extend(accumulator)

    combined_metrics = compute_opinion_metrics(
        truth_after=combined.truth_after,
        truth_before=combined.truth_before,
        pred_after=combined.pred_after,
        direction_tolerance=settings.controls.direction_tolerance,
    )
    write_metrics_json(out_dir / "combined_metrics.json", {"metrics": combined_metrics})
    return OpinionEvaluationResult(studies=results, combined_metrics=combined_metrics)

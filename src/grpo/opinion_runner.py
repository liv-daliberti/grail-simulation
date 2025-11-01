#!/usr/bin/env python
"""Evaluation runner for GRPO opinion regression."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Mapping, Sequence

from common.open_r1.example_utils import row_to_training_example
from common.opinion import (
    DEFAULT_SPECS,
    OpinionSpec,
    compute_opinion_metrics,
    format_opinion_user_prompt,
)
from common.pipeline.io import write_metrics_json

from .dataset import load_dataset_split
from .model import generate_chat_completion
from .opinion_types import (
    OpinionEvaluationResult,
    OpinionInferenceContext,
    OpinionPromptSettings,
    OpinionStudyContext,
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
    _CombinedVectors,
    _OpinionPredictionArtefact,
    _StudyAccumulator,
)
from .opinion_utils import (
    _compute_baseline_metrics,
    _extract_user_prompt,
    _parse_prediction,
    _select_examples_by_participant,
)
from .opinion_io import (
    _attempt_reuse_cached_result,
    _prepare_study_files,
    _persist_study_outputs,
    _resume_from_predictions_if_needed,
)

LOGGER = logging.getLogger("grpo.opinion")


def _direction_class(value: float, tol: float) -> int:
    """Classify direction by tolerance: -1 (down), 0 (flat), 1 (up)."""
    if abs(value) <= tol:
        return 0
    return 1 if value > 0 else -1


def _resolve_studies(include: Sequence[str] | None) -> Sequence[OpinionSpec]:
    """Return opinion specs filtered by ``include`` (if provided)."""

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


def _log_single_prediction(
    *,
    spec: OpinionSpec,
    idx: int,
    artefact: _OpinionPredictionArtefact,
    direction_tolerance: float,
) -> None:
    """Log detailed metrics for a single prediction and print a concise line."""

    messages = artefact.payload.get("messages", [])  # type: ignore[assignment]
    if not isinstance(messages, list):
        messages = []
    user_prompt = _extract_user_prompt(messages)

    before = float(artefact.before)
    after = float(artefact.after)
    pred = float(artefact.prediction)
    err = abs(pred - after)
    dir_ok = _direction_class(after - before, float(direction_tolerance)) == _direction_class(
        pred - before, float(direction_tolerance)
    )
    LOGGER.info(
        (
            "[OPINION][%s][%d] before=%.3f after=%.3f pred=%.3f "
            "abs_err=%.3f dir_ok=%s\nQuestion: %s\nAnswer: %s"
        ),
        spec.key,
        idx,
        before,
        after,
        pred,
        err,
        dir_ok,
        user_prompt,
        str(artefact.payload.get("raw_output", "")).strip(),
    )
    print(
        (
            f"[grpo.opinion] study={spec.key} ex={idx} "
            f"before={before:.2f} after={after:.2f} pred={pred:.2f} "
            f"abs_err={err:.2f} dir_ok={str(dir_ok).lower()}"
        ),
        flush=True,
    )


def _maybe_flush_partial(
    *,
    files: OpinionStudyFiles,
    accumulator: _StudyAccumulator,
    spec: OpinionSpec,
    direction_tolerance: float,
    count: int,
    flush_every: int,
) -> None:
    """Persist partial artefacts periodically for resilience and resumability."""

    if flush_every <= 0 or count % flush_every != 0:
        return
    try:
        snapshot = compute_opinion_metrics(
            truth_after=accumulator.truth_after,
            truth_before=accumulator.truth_before,
            pred_after=accumulator.pred_after,
            direction_tolerance=direction_tolerance,
        )
        baseline = _compute_baseline_metrics(
            truth_after=accumulator.truth_after,
            truth_before=accumulator.truth_before,
        )
        _persist_study_outputs(
            files,
            metrics=snapshot,
            baseline=baseline,
            accumulator=accumulator,
        )
        LOGGER.debug(
            "[OPINION] flushed partial artefacts study=%s at %d participants",
            spec.key,
            count,
        )
    except (OSError, ValueError, TypeError) as err:
        LOGGER.warning(
            "[OPINION] flush failed at study=%s count=%d; continuing (%s)",
            spec.key,
            count,
            err,
        )


def _finalize_study(
    *,
    files: OpinionStudyFiles,
    spec: OpinionSpec,
    accumulator: _StudyAccumulator,
    direction_tolerance: float,
) -> OpinionStudyResult:
    """Compute final metrics, persist artefacts, and build the study result."""

    metrics = compute_opinion_metrics(
        truth_after=accumulator.truth_after,
        truth_before=accumulator.truth_before,
        pred_after=accumulator.pred_after,
        direction_tolerance=direction_tolerance,
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
    return result


def _evaluate_study(
    *,
    spec: OpinionSpec,
    examples: Sequence[Mapping[str, object]],
    study_context: OpinionStudyContext,
    files: OpinionStudyFiles,
) -> tuple[OpinionStudyResult, _StudyAccumulator]:
    """Score ``examples`` for ``spec`` and materialise study artefacts."""

    # Fast path: reuse cached metrics when available and allowed.
    reused = _attempt_reuse_cached_result(
        spec=spec, files=files, overwrite=study_context.overwrite
    )
    if reused is not None:
        return reused

    # Resume from predictions when present but metrics are missing.
    accumulator, processed = _resume_from_predictions_if_needed(files=files, spec=spec)

    flush_every = int(getattr(study_context, "flush_every", 0) or 0)
    for idx, example in enumerate(examples, start=1):
        if processed and idx <= processed:
            continue
        artefact = _score_opinion_example(
            example=example,
            spec=spec,
            prompts=study_context.prompts,
            context=study_context.inference,
        )
        if artefact is None:
            continue
        accumulator.record(artefact)
        _log_single_prediction(
            spec=spec,
            idx=idx,
            artefact=artefact,
            direction_tolerance=study_context.direction_tolerance,
        )
        _maybe_flush_partial(
            files=files,
            accumulator=accumulator,
            spec=spec,
            direction_tolerance=study_context.direction_tolerance,
            count=processed + idx,
            flush_every=flush_every,
        )

    result = _finalize_study(
        files=files,
        spec=spec,
        accumulator=accumulator,
        direction_tolerance=study_context.direction_tolerance,
    )
    return result, accumulator


def _score_opinion_example(
    *,
    example: Mapping[str, object],
    spec: OpinionSpec,
    prompts: OpinionPromptSettings,
    context: OpinionInferenceContext,
) -> _OpinionPredictionArtefact | None:
    """Return prediction artefacts for a single participant example."""

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
    settings: 'OpinionEvaluationSettings',
    out_dir: Path,
) -> OpinionEvaluationResult:
    """Evaluate opinion regression across configured studies."""

    LOGGER.info(
        (
            "[OPINION] loading dataset name=%s split=%s include_studies=%s "
            "max_participants=%s overwrite=%s"
        ),
        settings.dataset.name,
        settings.dataset.split,
        ",".join(settings.include_studies or ()) or "<all>",
        settings.controls.max_participants or "<no-cap>",
        settings.controls.overwrite,
    )
    include_text = ",".join(settings.include_studies or ()) or "<all>"
    max_participants_text = settings.controls.max_participants or "<no-cap>"
    print(
        (
            f"[grpo.opinion] dataset={settings.dataset.name} "
            f"split={settings.dataset.split} include_studies={include_text} "
            f"max_participants={max_participants_text} "
            f"overwrite={settings.controls.overwrite}"
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
        overwrite=settings.controls.overwrite,
        flush_every=int(getattr(settings.controls, "flush_every", 0) or 0),
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


__all__ = [
    "run_opinion_evaluation",
]

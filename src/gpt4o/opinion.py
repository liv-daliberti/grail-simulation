#!/usr/bin/env python
# pylint: disable=too-many-lines
"""Opinion-shift evaluation for the GPT-4o baseline."""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, IO, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np

from common.data import hf_datasets as _hf_datasets
from common.opinion import DEFAULT_SPECS, OpinionSpec, compute_opinion_metrics

from .config import DATASET_NAME, EVAL_SPLIT, OPINION_SYSTEM_PROMPT
from .utils import (
    InvocationParams,
    RetryPolicy,
    call_gpt4o_with_retries,
    qa_log_path_for,
)

LOGGER = logging.getLogger("gpt4o.opinion")

DOWNLOAD_CONFIG_CLS, LOAD_DATASET, LOAD_FROM_DISK = _hf_datasets.get_dataset_loaders()

_ANSWER_PATTERN = re.compile(
    r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True)
class OpinionArtifacts:
    """Filesystem artefacts generated for a single study."""

    metrics: Path
    predictions: Path
    qa_log: Path


@dataclass(frozen=True)
class OpinionMetricBundle:
    """Pair the computed metrics and baseline payloads."""

    metrics: Mapping[str, object]
    baseline: Mapping[str, object]


@dataclass(frozen=True)
class OpinionStudyResult:
    """Container capturing metrics and artefact paths for one opinion study."""

    study_key: str
    study_label: str
    issue: str
    participants: int
    eligible: int
    artifacts: OpinionArtifacts
    metric_bundle: OpinionMetricBundle

    @property
    def metrics(self) -> Mapping[str, object]:
        """Return the computed metrics payload."""
        return self.metric_bundle.metrics

    @property
    def baseline(self) -> Mapping[str, object]:
        """Return the baseline metrics payload."""
        return self.metric_bundle.baseline

    @property
    def metrics_path(self) -> Path:
        """Expose the metrics artefact path for downstream consumers."""
        return self.artifacts.metrics

    @property
    def predictions_path(self) -> Path:
        """Expose the predictions artefact path for downstream consumers."""
        return self.artifacts.predictions

    @property
    def qa_log_path(self) -> Path:
        """Expose the QA log artefact path for downstream consumers."""
        return self.artifacts.qa_log


@dataclass(frozen=True)
class OpinionEvaluationResult:
    """Aggregate payload returned by the GPT-4o opinion evaluator."""

    studies: Mapping[str, OpinionStudyResult]
    combined_metrics: Mapping[str, object]
    config_label: str


@dataclass(frozen=True)
class OpinionFilters:
    """Filter configuration applied prior to evaluation."""

    issues: set[str]
    studies: set[str]

    def allows(self, issue: str, study: str) -> bool:
        """Return ``True`` when ``issue``/``study`` pass the configured filters."""
        issue_key = issue.lower().strip() if issue else ""
        study_key = study.lower().strip() if study else ""
        if self.issues and issue_key not in self.issues:
            return False
        if self.studies and study_key not in self.studies:
            return False
        return True


@dataclass(frozen=True)
class OpinionRuntime:
    """Runtime configuration for GPT-4o opinion inference."""

    temperature: float
    max_tokens: int
    top_p: float | None
    deployment: str | None
    retries: int
    retry_delay: float


@dataclass(frozen=True)
class OpinionLimits:
    """Execution limits and flags applied during evaluation."""

    eval_max: int
    direction_tolerance: float
    overwrite: bool


@dataclass(frozen=True)
class OpinionSettings:
    """Resolved configuration shared across the evaluation run."""

    dataset_name: str
    cache_dir: str | None
    filters: OpinionFilters
    requested_specs: Sequence[str]
    limits: OpinionLimits
    runtime: OpinionRuntime


@dataclass
class CombinedAccumulator:
    """Accumulate per-study metrics for the combined summary."""

    truth_before: List[float]
    truth_after: List[float]
    pred_after: List[float]

    def extend(
        self,
        truth_before: Sequence[float],
        truth_after: Sequence[float],
        pred_after: Sequence[float],
    ) -> None:
        """Extend the accumulator with additional study-level vectors."""
        self.truth_before.extend(truth_before)
        self.truth_after.extend(truth_after)
        self.pred_after.extend(pred_after)

    def compute_metrics(self, direction_tolerance: float) -> Mapping[str, object]:
        """Return combined metrics using the accumulated vectors."""
        if not self.truth_after:
            return {}
        return compute_opinion_metrics(
            truth_after=self.truth_after,
            truth_before=self.truth_before,
            pred_after=self.pred_after,
            direction_tolerance=direction_tolerance,
        )


@dataclass
class StudyPredictionBatch:
    """Capture inference artefacts for a single study evaluation."""

    payloads: List[Mapping[str, object]]
    truth_before: List[float]
    truth_after: List[float]
    pred_after: List[float]


@dataclass(frozen=True)
class StudyMetricsPayload:
    """Bundle metrics and participant counts for artefact generation."""

    participants: int
    metrics: Mapping[str, object]
    baseline: Mapping[str, object]


@dataclass(frozen=True)
class CachedStudyPayload:
    """Cached metrics payload reconstructed from disk."""

    participants: int
    metrics: Mapping[str, object]
    baseline: Mapping[str, object]
    study_label: str
    issue: str


@dataclass(frozen=True)
class CachedPredictionVectors:
    """Cached prediction vectors reconstructed from disk."""

    truth_before: List[float]
    truth_after: List[float]
    pred_after: List[float]


@dataclass(frozen=True)
class QALogEntry:
    """Payload describing a single QA log record."""

    idx: int
    spec: OpinionSpec
    messages: Sequence[Mapping[str, object]]
    raw_output: str


@dataclass(frozen=True)
class ExampleProcessingContext:
    """Resources shared by per-example processing helpers."""

    qa_log: IO[str]
    batch: StudyPredictionBatch


def _parse_tokens(raw: str) -> Tuple[List[str], set[str]]:
    """Return the requested token list and a lowercase set for fast membership tests.

    :param raw: Comma-separated input provided via CLI options.
    :returns: Tuple of the ordered tokens and their lowercase variants.
    :rtype: tuple[list[str], set[str]]
    """
    tokens: List[str] = []
    for segment in (raw or "").split(","):
        candidate = segment.strip()
        if candidate:
            tokens.append(candidate)
    lowered = {token.lower() for token in tokens}
    if "all" in lowered:
        lowered.clear()
    return tokens, lowered


def _float_or_none(value: object) -> Optional[float]:
    """Return ``value`` parsed as a finite float when possible.

    :param value: Raw numeric-like value drawn from the dataset.
    :returns: Parsed float or ``None`` when conversion fails.
    """

    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric):
        return None
    return numeric


def _document_from_example(example: Mapping[str, object]) -> str:
    """Assemble the viewer profile/state text bundle used for opinion prompts.

    :param example: Participant entry containing profile and context fields.
    :returns: Normalised multi-section document string.
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


def _clip_prediction(value: float) -> float:
    """Clamp predictions to the 1–7 opinion index range.

    :param value: Raw predicted opinion index.
    :returns: Prediction bounded to the inclusive ``[1, 7]`` range.
    """

    return max(1.0, min(7.0, float(value)))


def _baseline_metrics(
    truth_before: Sequence[float], truth_after: Sequence[float]
) -> Dict[str, object]:
    """Compute baseline metrics mirroring the KNN/XGB implementations.

    :param truth_before: Ground-truth pre-study opinion indices.
    :param truth_after: Ground-truth post-study opinion indices.
    :returns: Dictionary of baseline statistics used in reports.
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


def _load_materialised_split(
    dataset: object, preferred_split: str
) -> Tuple[str, Iterable[Mapping[str, object]]]:
    """Return the evaluation split from either a DatasetDict or single dataset.

    :param dataset: Materialised dataset or dataset dictionary.
    :param preferred_split: Primary split name requested for evaluation.
    :returns: Tuple of chosen split name and iterable rows.
    :raises RuntimeError: If ``dataset`` is missing.
    """

    if dataset is None:
        raise RuntimeError("Expected dataset to be materialised before opinion evaluation.")

    eval_split = preferred_split
    available: List[str] = []
    if hasattr(dataset, "keys"):
        try:
            available = list(dataset.keys())  # type: ignore[attr-defined]
        except Exception:  # pylint: disable=broad-except
            available = []
    if available:
        for candidate in (preferred_split, "validation", "eval", "test", "train"):
            if candidate in available:
                eval_split = candidate
                break
        else:
            eval_split = available[0]
        split_dataset = dataset[eval_split]  # type: ignore[index]
    else:
        split_dataset = dataset
    return eval_split, split_dataset


def _resolve_spec_keys(raw: str | None) -> List[str]:
    """Return the ordered opinion study keys to evaluate.

    :param raw: Comma-separated opinion study key overrides.
    :returns: Ordered list of study keys to evaluate.
    """

    if not raw:
        return [spec.key for spec in DEFAULT_SPECS]
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        return [spec.key for spec in DEFAULT_SPECS]
    return tokens


def _settings_from_args(args) -> OpinionSettings:
    """Construct :class:`OpinionSettings` from CLI or programmatic arguments."""

    dataset_name = str(getattr(args, "dataset", "") or DATASET_NAME)
    cache_dir = getattr(args, "cache_dir", None)

    issues_raw = str(getattr(args, "issues", "") or "")
    studies_raw = str(getattr(args, "studies", "") or "")
    _, issue_filter = _parse_tokens(issues_raw)
    _, study_filter = _parse_tokens(studies_raw)
    filters = OpinionFilters(issues=issue_filter, studies=study_filter)

    requested_specs = _resolve_spec_keys(getattr(args, "opinion_studies", None))

    eval_max = getattr(args, "opinion_max_participants", 0) or getattr(args, "eval_max", 0)
    direction_tolerance = float(
        getattr(
            args,
            "opinion_direction_tolerance",
            getattr(args, "direction_tolerance", 1e-6),
        )
    )
    overwrite = bool(getattr(args, "overwrite", False))
    limits = OpinionLimits(
        eval_max=int(eval_max or 0),
        direction_tolerance=direction_tolerance,
        overwrite=overwrite,
    )

    runtime = OpinionRuntime(
        temperature=float(getattr(args, "temperature", 0.0)),
        max_tokens=int(getattr(args, "max_tokens", 32)),
        top_p=getattr(args, "top_p", None),
        deployment=getattr(args, "deployment", None),
        retries=max(1, int(getattr(args, "request_retries", 5) or 5)),
        retry_delay=max(0.0, float(getattr(args, "request_retry_delay", 1.0) or 0.0)),
    )

    return OpinionSettings(
        dataset_name=dataset_name,
        cache_dir=cache_dir,
        filters=filters,
        requested_specs=requested_specs,
        limits=limits,
        runtime=runtime,
    )


class OpinionEvaluationRunner:
    """Stateful helper running GPT-4o opinion evaluations."""

    def __init__(
        self,
        args,
        *,
        config_label: str,
        out_dir: Path,
    ) -> None:
        """Initialise the runner with CLI arguments and derived configuration.

        :param args: Namespace generated by the GPT-4o CLI parser.
        :param config_label: Human-readable identifier for the sweep configuration.
        :param out_dir: Directory where evaluation artefacts should be written.
        """
        self._config_label = config_label
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._settings = _settings_from_args(args)

    @property
    def settings(self) -> OpinionSettings:
        """Return the resolved evaluation settings."""
        return self._settings

    @property
    def runtime(self) -> OpinionRuntime:
        """Return the runtime invocation parameters."""
        return self._settings.runtime

    @property
    def limits(self) -> OpinionLimits:
        """Return the execution limits associated with the run."""
        return self._settings.limits

    @property
    def out_dir(self) -> Path:
        """Return the root output directory for evaluation artefacts."""
        return self._out_dir

    def _allows(self, issue: str, study: str) -> bool:
        """Return ``True`` when the current filters allow the example.

        :param issue: Issue label from the dataset row.
        :param study: Participant study identifier.
        :returns: Whether the example passes the configured filters.
        :rtype: bool
        """
        return self.settings.filters.allows(issue, study)

    def _load_dataset(self):
        """Load the opinion dataset from disk or the Hugging Face hub.

        :returns: List of materialised dataset rows for evaluation.
        :raises RuntimeError: If the evaluation split cannot be determined.
        :rtype: list[Mapping[str, object]]
        """
        dataset_path = Path(self.settings.dataset_name)
        dataset = None

        if dataset_path.exists():
            _hf_datasets.require_dataset_support(needs_local=True)
            assert LOAD_FROM_DISK is not None  # narrow for mypy
            LOGGER.info("Loading opinion dataset from disk: %s", dataset_path)
            dataset = LOAD_FROM_DISK(str(dataset_path))  # type: ignore[arg-type]
        else:
            _hf_datasets.require_dataset_support()
            assert LOAD_DATASET is not None and DOWNLOAD_CONFIG_CLS is not None
            download_config = DOWNLOAD_CONFIG_CLS(  # type: ignore[misc]
                resume_download=True,
                max_retries=2,
            )
            LOGGER.info("Downloading opinion dataset %s", self.settings.dataset_name)
            dataset = LOAD_DATASET(  # type: ignore[misc]
                self.settings.dataset_name,
                cache_dir=self.settings.cache_dir,
                download_config=download_config,
            )
        split_name, split_dataset = _load_materialised_split(dataset, EVAL_SPLIT)
        LOGGER.info("Using opinion evaluation split '%s'.", split_name)
        return list(split_dataset)

    def _collect_examples(
        self,
        rows: Sequence[Mapping[str, object]],
        spec: OpinionSpec,
    ) -> List[Mapping[str, object]]:
        """Return filtered participant examples for the provided study spec.

        :param rows: Entire dataset rows for the active split.
        :param spec: Opinion study definition that controls filtering.
        :returns: Ordered list of participant payloads ready for prompting.
        :rtype: list[Mapping[str, object]]
        """
        per_participant: MutableMapping[str, Tuple[int, Mapping[str, object]]] = {}
        retained: List[Mapping[str, object]] = []

        for entry in rows:
            issue = str(entry.get("issue") or "").strip()
            study = str(entry.get("participant_study") or "").strip()
            if not self._allows(issue, study):
                continue
            if study.lower() != spec.key.lower():
                continue
            before = _float_or_none(entry.get(spec.before_column))
            after = _float_or_none(entry.get(spec.after_column))
            if before is None or after is None:
                continue
            participant_id = str(entry.get("participant_id") or "").strip()
            if not participant_id:
                continue
            document = _document_from_example(entry)
            if not document:
                continue
            try:
                step_index = int(entry.get("step_index") or -1)
            except (TypeError, ValueError):
                step_index = -1

            payload = {
                "participant_id": participant_id,
                "document": document,
                "before": before,
                "after": after,
                "issue": issue,
                "study": study,
                "step_index": step_index,
                "raw": entry,
            }
            existing = per_participant.get(participant_id)
            if existing is None or step_index >= existing[0]:
                per_participant[participant_id] = (step_index, payload)

        for _, payload in per_participant.values():
            retained.append(payload)

        retained.sort(key=lambda item: (item["participant_id"], item["step_index"]))
        if self.limits.eval_max and len(retained) > self.limits.eval_max:
            LOGGER.info(
                "[OPINION] Limiting study=%s participants to %d (from %d).",
                spec.key,
                self.limits.eval_max,
                len(retained),
            )
            retained = retained[: self.limits.eval_max]
        return retained

    def _build_messages(
        self,
        spec: OpinionSpec,
        example: Mapping[str, object],
    ) -> List[Mapping[str, str]]:
        """Construct the system/user messages for a single participant example.

        :param spec: Opinion study metadata driving prompt phrasing.
        :param example: Participant bundle produced by ``_collect_examples``.
        :returns: Messages compatible with the chat completion API.
        :rtype: list[dict[str, str]]
        """
        before = example["before"]
        issue_label = spec.issue.replace("_", " ").title()
        document = example["document"]

        user_lines = [
            f"Issue: {issue_label}",
            "Opinion scale: 1 = strongly oppose, 7 = strongly support.",
            f"Pre-study opinion index: {before:.2f}",
            "",
            "Viewer context:",
            document,
            "",
            (
                "After the participant watches the specified next video, "
                "estimate their post-study opinion index."
            ),
            (
                "Reason briefly inside <think> then output ONLY the numeric "
                "index (1-7) inside <answer>."
            ),
            "Do not include labels, punctuation, or words outside the required tags.",
        ]

        return [
            {"role": "system", "content": OPINION_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(user_lines).strip()},
        ]

    def _infer_prediction(self, messages: List[Mapping[str, str]]) -> Tuple[float, str]:
        """Call GPT-4o and parse the predicted opinion index.

        :param messages: Chat completion payload constructed for the participant.
        :returns: Tuple containing the clipped numeric prediction and raw output text.
        :rtype: tuple[float, str]
        """
        invocation = InvocationParams(
            max_tokens=self.runtime.max_tokens,
            temperature=self.runtime.temperature,
            top_p=self.runtime.top_p,
            deployment=self.runtime.deployment,
        )
        policy = RetryPolicy(
            attempts=self.runtime.retries,
            delay=self.runtime.retry_delay,
        )
        try:
            raw_output = call_gpt4o_with_retries(
                messages,
                invocation=invocation,
                retry=policy,
                logger=LOGGER,
            )
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception(
                "Opinion evaluation call failed after %d attempts: %s",
                policy.attempts,
                exc,
            )
            return float("nan"), f"(error after {policy.attempts} attempts: {exc})"

        match = _ANSWER_PATTERN.search(raw_output)
        if match:
            numeric = match.group(1)
        else:
            match = _NUMBER_PATTERN.search(raw_output)
            numeric = match.group(0) if match else ""
        try:
            value = float(numeric)
        except (TypeError, ValueError):
            LOGGER.warning("Unable to parse opinion prediction from output: %r", raw_output)
            return float("nan"), raw_output

        return _clip_prediction(value), raw_output

    @staticmethod
    def _extract_log_messages(messages: Sequence[Mapping[str, str]]) -> Tuple[str, str]:
        """Return the system prompt and latest user message for QA logging."""
        system_prompt = ""
        for message in messages:
            if (
                isinstance(message, Mapping)
                and message.get("role") == "system"
                and message.get("content")
            ):
                system_prompt = str(message["content"]).strip()
                break
        user_prompt = ""
        for message in reversed(messages):
            if (
                isinstance(message, Mapping)
                and message.get("role") == "user"
                and message.get("content")
            ):
                user_prompt = str(message["content"]).strip()
                break
        return system_prompt, user_prompt

    def _artifacts_for_spec(self, spec: OpinionSpec) -> OpinionArtifacts:
        """Return the artefact paths associated with ``spec``."""
        study_dir = self.out_dir / spec.key
        study_dir.mkdir(parents=True, exist_ok=True)
        return OpinionArtifacts(
            metrics=study_dir / "metrics.json",
            predictions=study_dir / "predictions.jsonl",
            qa_log=qa_log_path_for(study_dir),
        )

    def _write_predictions(
        self,
        artifacts: OpinionArtifacts,
        predictions: Sequence[Mapping[str, object]],
    ) -> None:
        """Persist predictions to disk."""
        with artifacts.predictions.open("w", encoding="utf-8") as handle:
            for payload in predictions:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    @staticmethod
    def _load_cached_metrics_payload(
        metrics_path: Path,
        *,
        study_key: str,
        default_label: str,
        default_issue: str,
    ) -> CachedStudyPayload | None:
        """Return cached study metrics when available and well-formed."""
        try:
            with metrics_path.open("r", encoding="utf-8") as handle:
                cached_payload = json.load(handle)
        except (OSError, json.JSONDecodeError):
            LOGGER.warning(
                "[OPINION] Cached metrics malformed for study=%s; recomputing.",
                study_key,
            )
            return None

        metrics = cached_payload.get("metrics", {}) or {}
        baseline = cached_payload.get("baseline", {}) or {}
        participants = int(cached_payload.get("participants", 0) or 0)
        study_label = str(cached_payload.get("study_label", default_label or study_key))
        issue = str(cached_payload.get("issue", default_issue))
        return CachedStudyPayload(
            participants=participants,
            metrics=metrics,
            baseline=baseline,
            study_label=study_label,
            issue=issue,
        )

    @staticmethod
    def load_cached_prediction_vectors(
        predictions_path: Path,
        *,
        study_key: str,
    ) -> CachedPredictionVectors | None:
        """Return cached prediction vectors when available and well-formed."""
        truth_before: List[float] = []
        truth_after: List[float] = []
        pred_after: List[float] = []
        try:
            with predictions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    try:
                        truth_before.append(float(row.get("before")))
                        truth_after.append(float(row.get("after")))
                        pred_after.append(float(row.get("predicted_after")))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            LOGGER.warning(
                "[OPINION] Unable to read cached predictions for study=%s; recomputing.",
                study_key,
            )
            return None
        return CachedPredictionVectors(
            truth_before=truth_before,
            truth_after=truth_after,
            pred_after=pred_after,
        )

    def _write_metrics_file(
        self,
        spec: OpinionSpec,
        artifacts: OpinionArtifacts,
        payload: StudyMetricsPayload,
    ) -> None:
        """Write the metrics JSON payload for ``spec``."""
        content = {
            "study": spec.key,
            "issue": spec.issue,
            "study_label": spec.label,
            "participants": payload.participants,
            "metrics": payload.metrics,
            "baseline": payload.baseline,
            "config": {
                "temperature": self.runtime.temperature,
                "max_tokens": self.runtime.max_tokens,
                "top_p": self.runtime.top_p,
            },
        }
        with artifacts.metrics.open("w", encoding="utf-8") as handle:
            json.dump(content, handle, ensure_ascii=False, indent=2)

    def _load_cached_result(
        self,
        spec: OpinionSpec,
        artifacts: OpinionArtifacts,
        accumulator: CombinedAccumulator,
    ) -> OpinionStudyResult | None:
        """Load cached artefacts when reuse is permitted."""
        if not artifacts.metrics.exists() or not artifacts.predictions.exists():
            return None
        metrics_payload = self._load_cached_metrics_payload(
            artifacts.metrics,
            study_key=spec.key,
            default_label=spec.label,
            default_issue=spec.issue,
        )
        if metrics_payload is None:
            return None

        vectors = self.load_cached_prediction_vectors(
            artifacts.predictions,
            study_key=spec.key,
        )
        if vectors is None:
            return None

        accumulator.extend(vectors.truth_before, vectors.truth_after, vectors.pred_after)
        eligible = int(metrics_payload.metrics.get("eligible", len(vectors.truth_after)))
        bundle = OpinionMetricBundle(
            metrics=metrics_payload.metrics,
            baseline=metrics_payload.baseline,
        )
        participants = metrics_payload.participants or len(vectors.truth_after)
        LOGGER.info(
            "[OPINION][SKIP] study=%s (metrics cached).",
            spec.key,
        )
        return OpinionStudyResult(
            study_key=spec.key,
            study_label=metrics_payload.study_label or spec.label,
            issue=metrics_payload.issue or spec.issue,
            participants=participants,
            eligible=eligible,
            artifacts=artifacts,
            metric_bundle=bundle,
        )

    def _write_qa_log_entry(self, qa_log: IO[str], entry: QALogEntry) -> None:
        """Write a single QA log entry for the participant."""
        system_prompt, question = self._extract_log_messages(entry.messages)
        qa_log.write(f"### Participant {entry.idx}\n")
        qa_log.write(f"Study: {entry.spec.key} | Issue: {entry.spec.issue}\n")
        qa_log.write("SYSTEM:\n")
        qa_log.write(f"{system_prompt}\n")
        qa_log.write("QUESTION:\n")
        qa_log.write(f"{question}\n")
        qa_log.write("ANSWER:\n")
        qa_log.write(f"{entry.raw_output.strip()}\n\n")
        qa_log.flush()

    def _process_example(
        self,
        *,
        spec: OpinionSpec,
        example: Mapping[str, object],
        idx: int,
        context: ExampleProcessingContext,
    ) -> None:
        """Run inference for a single participant example and update artefacts."""
        messages = self._build_messages(spec, example)
        prediction, raw_output = self._infer_prediction(messages)
        if math.isnan(prediction):
            prediction = example["before"]
        prediction = _clip_prediction(prediction)

        context.batch.payloads.append(
            {
                "participant_id": example["participant_id"],
                "study": spec.key,
                "issue": spec.issue,
                "before": example["before"],
                "after": example["after"],
                "predicted_after": prediction,
                "messages": messages,
                "raw_output": raw_output,
            }
        )
        context.batch.truth_before.append(float(example["before"]))
        context.batch.truth_after.append(float(example["after"]))
        context.batch.pred_after.append(float(prediction))

        self._write_qa_log_entry(
            context.qa_log,
            QALogEntry(
                idx=idx,
                spec=spec,
                messages=messages,
                raw_output=raw_output,
            ),
        )

    def _gather_predictions(
        self,
        spec: OpinionSpec,
        examples: Sequence[Mapping[str, object]],
        qa_log_path: Path,
    ) -> StudyPredictionBatch:
        """Invoke GPT-4o for ``examples`` and capture artefacts."""
        batch = StudyPredictionBatch(
            payloads=[],
            truth_before=[],
            truth_after=[],
            pred_after=[],
        )
        with qa_log_path.open("w", encoding="utf-8") as qa_log:
            processing_context = ExampleProcessingContext(qa_log=qa_log, batch=batch)
            for idx, example in enumerate(examples, start=1):
                self._process_example(
                    spec=spec,
                    example=example,
                    idx=idx,
                    context=processing_context,
                )

        return batch

    def _evaluate_study(
        self,
        spec: OpinionSpec,
        dataset_rows: Sequence[Mapping[str, object]],
        accumulator: CombinedAccumulator,
    ) -> OpinionStudyResult | None:
        """Evaluate ``spec`` and update the combined accumulator."""
        examples = self._collect_examples(dataset_rows, spec)
        if not examples:
            LOGGER.warning("[OPINION] No eligible participants for study=%s.", spec.key)
            return None

        artifacts = self._artifacts_for_spec(spec)
        if not self.limits.overwrite:
            cached = self._load_cached_result(spec, artifacts, accumulator)
            if cached is not None:
                return cached

        batch = self._gather_predictions(spec, examples, artifacts.qa_log)
        metrics = compute_opinion_metrics(
            truth_after=batch.truth_after,
            truth_before=batch.truth_before,
            pred_after=batch.pred_after,
            direction_tolerance=self.limits.direction_tolerance,
        )
        baseline = _baseline_metrics(batch.truth_before, batch.truth_after)

        self._write_predictions(artifacts, batch.payloads)
        self._write_metrics_file(
            spec,
            artifacts,
            StudyMetricsPayload(
                participants=len(examples),
                metrics=metrics,
                baseline=baseline,
            ),
        )

        accumulator.extend(batch.truth_before, batch.truth_after, batch.pred_after)
        return OpinionStudyResult(
            study_key=spec.key,
            study_label=spec.label,
            issue=spec.issue,
            participants=len(examples),
            eligible=int(metrics.get("eligible", 0)),
            artifacts=artifacts,
            metric_bundle=OpinionMetricBundle(metrics=metrics, baseline=baseline),
        )

    def run(self) -> OpinionEvaluationResult:
        """Execute the opinion evaluation for the configured studies."""
        dataset_rows = self._load_dataset()
        specs_by_key = {spec.key.lower(): spec for spec in DEFAULT_SPECS}
        accumulator = CombinedAccumulator([], [], [])
        studies: Dict[str, OpinionStudyResult] = {}

        for key in self.settings.requested_specs:
            spec = specs_by_key.get(key.lower())
            if spec is None:
                LOGGER.warning("Skipping unknown opinion study '%s'.", key)
                continue

            result = self._evaluate_study(spec, dataset_rows, accumulator)
            if result is not None:
                LOGGER.info(
                    "[OPINION] study=%s participants=%d mae_after=%.3f direction_acc=%s",
                    spec.key,
                    result.participants,
                    result.metrics.get("mae_after", float("nan")),
                    result.metrics.get("direction_accuracy"),
                )
                studies[spec.key] = result

        combined_metrics = accumulator.compute_metrics(self.limits.direction_tolerance)
        return OpinionEvaluationResult(
            studies=studies,
            combined_metrics=combined_metrics,
            config_label=self._config_label,
        )


def run_opinion_evaluations(
    *,
    args,
    config_label: str,
    out_dir: Path,
) -> OpinionEvaluationResult:
    """
    Execute GPT-4o opinion evaluations for the requested studies.

    :param args: Parsed CLI arguments shared with the main pipeline.
    :param config_label: Identifier for the active temperature/top-p/max-token bundle.
    :param out_dir: Directory receiving opinion predictions and metrics.
    :returns: Aggregated evaluation result suitable for report generation.
    """

    runner = OpinionEvaluationRunner(args, config_label=config_label, out_dir=out_dir)
    return runner.run()


__all__ = [
    "OpinionEvaluationResult",
    "OpinionStudyResult",
    "run_opinion_evaluations",
]

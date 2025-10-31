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

"""GPT-4o opinion evaluation runner."""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, cast

from importlib import import_module
from ..config import EVAL_SPLIT, OPINION_SYSTEM_PROMPT
from ..utils import (
    InvocationParams,
    RetryPolicy,
    qa_log_path_for,
    try_call_with_policy,
)
from .helpers import (
    baseline_metrics,
    clip_prediction,
    collect_examples,
    CollectExamplesConfig,
    load_materialised_split,
)
from .models import (
    CachedPredictionVectors,
    CachedStudyPayload,
    CombinedAccumulator,
    ExampleProcessingContext,
    OpinionArtifacts,
    OpinionEvaluationResult,
    OpinionSettings,
    OpinionStudyResult,
    QALogEntry,
    StudyMetricsPayload,
    StudyPredictionBatch,
)
from .settings import build_settings
_hf_datasets = import_module("common.data.hf_datasets")
_common_opinion = import_module("common.opinion")
DEFAULT_SPECS = _common_opinion.DEFAULT_SPECS
OpinionSpec = _common_opinion.OpinionSpec
compute_opinion_metrics = _common_opinion.compute_opinion_metrics
format_opinion_user_prompt = _common_opinion.format_opinion_user_prompt


LOGGER = logging.getLogger("gpt4o.opinion")

DOWNLOAD_CONFIG_CLS, LOAD_DATASET, LOAD_FROM_DISK = (
    cast(tuple[Any, Callable[..., Any], Callable[[str], Any]], _hf_datasets.get_dataset_loaders())
)

_ANSWER_PATTERN = re.compile(
    r"<answer>\s*([-+]?\d+(?:\.\d+)?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)
_NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:\.\d+)?")


class OpinionEvaluationRunner:
    """Stateful helper running GPT-4o opinion evaluations.

    Orchestrates dataset loading, participant filtering, GPT-4o inference,
    artefact writing, and per-study/combined metric computation.
    """

    def __init__(
        self,
        args,
        *,
        config_label: str,
        out_dir: Path,
    ) -> None:
        """Initialise the runner with CLI arguments and derived configuration.

        :param args: CLI namespace or argument container controlling evaluation.
        :param config_label: Human-readable key for the selected configuration.
        :param out_dir: Root directory where artefacts will be written.
        :returns: ``None``.
        """
        self._config_label = config_label
        self._out_dir = out_dir
        self._out_dir.mkdir(parents=True, exist_ok=True)
        self._settings = build_settings(args)

    @property
    def settings(self) -> OpinionSettings:
        """Return the resolved evaluation settings.

        :returns: Immutable :class:`OpinionSettings` shared by the run.
        :rtype: OpinionSettings
        """
        return self._settings

    @property
    def runtime(self):
        """Return the runtime invocation parameters.

        :returns: :class:`OpinionRuntime` describing temperature, tokens, etc.
        """
        return self._settings.runtime

    @property
    def limits(self):
        """Return the execution limits associated with the run.

        :returns: :class:`OpinionLimits` instance including eval caps and flags.
        """
        return self._settings.limits

    @property
    def out_dir(self) -> Path:
        """Return the root output directory for evaluation artefacts.

        :returns: Filesystem path for per-study outputs and QA logs.
        :rtype: pathlib.Path
        """
        return self._out_dir

    def _allows(self, issue: str, study: str) -> bool:
        """Return ``True`` when the current filters allow the example.

        :param issue: Issue label for the dataset row.
        :param study: Participant study key for the dataset row.
        :returns: ``True`` if both fields pass the active filters.
        :rtype: bool
        """
        return self.settings.filters.allows(issue, study)

    def _load_dataset(self) -> List[Mapping[str, object]]:
        """Load the opinion dataset from disk or the Hugging Face hub.

        :returns: Materialised rows for the evaluation split.
        :rtype: list[Mapping[str, object]]
        """
        dataset_path = Path(self.settings.dataset_name)
        dataset = None

        if dataset_path.exists():
            _hf_datasets.require_dataset_support(needs_local=True)
            assert LOAD_FROM_DISK is not None  # narrow for mypy
            LOGGER.info("Loading opinion dataset from disk: %s", dataset_path)
            dataset = LOAD_FROM_DISK(str(dataset_path))
        else:
            _hf_datasets.require_dataset_support()
            assert LOAD_DATASET is not None and DOWNLOAD_CONFIG_CLS is not None
            download_config = DOWNLOAD_CONFIG_CLS(
                resume_download=True,
                max_retries=2,
            )
            LOGGER.info("Downloading opinion dataset %s", self.settings.dataset_name)
            dataset = LOAD_DATASET(
                self.settings.dataset_name,
                cache_dir=self.settings.cache_dir,
                download_config=download_config,
            )
        split_name, split_dataset = load_materialised_split(dataset, EVAL_SPLIT)
        LOGGER.info("Using opinion evaluation split '%s'.", split_name)
        return list(split_dataset)

    def _collect_examples(
        self,
        rows: Sequence[Mapping[str, object]],
        spec: OpinionSpec,
    ) -> List[Mapping[str, object]]:
        """Return filtered participant examples for the provided study spec.

        :param rows: Materialised dataset rows for the evaluation split.
        :param spec: :class:`~common.opinion.OpinionSpec` defining the study.
        :returns: Canonical participant payloads after filtering/deduplication.
        :rtype: list[Mapping[str, object]]
        """
        config = CollectExamplesConfig(
            allows=self._allows,
            eval_max=self.limits.eval_max,
        )
        retained, original_count = collect_examples(rows, spec, config)
        if (
            self.limits.eval_max
            and original_count > self.limits.eval_max
            and len(retained) == self.limits.eval_max
        ):
            LOGGER.info(
                "[OPINION] Limiting study=%s participants to %d (from %d).",
                spec.key,
                self.limits.eval_max,
                original_count,
            )
        return retained

    def _build_messages(
        self,
        spec: OpinionSpec,
        example: Mapping[str, object],
    ) -> List[Mapping[str, str]]:
        """Construct the system/user messages for a single participant example.

        :param spec: Study specification providing ``issue`` and label.
        :param example: Canonical participant payload from :func:`collect_examples`.
        :returns: Sequence with one system and one user message.
        :rtype: list[Mapping[str, str]]
        """
        before = example["before"]
        issue_label = spec.issue.replace("_", " ").title()
        document = example["document"]

        user_message = format_opinion_user_prompt(
            issue_label=issue_label,
            pre_study_index=before,
            viewer_context=document,
            post_watch_instruction=(
                "After the participant watches the specified next video, "
                "estimate their post-study opinion index."
            ),
            extra_lines=(
                "Do not include labels, punctuation, or words outside the required tags.",
            ),
        )

        return [
            {"role": "system", "content": OPINION_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

    def _infer_prediction(self, messages: List[Mapping[str, str]]) -> Tuple[float, str]:
        """Call GPT-4o and parse the predicted opinion index.

        :param messages: Chat completion payload to send to GPT-4o.
        :returns: Tuple of ``(predicted_after, raw_output)`` where
            ``predicted_after`` is clipped to the 1–7 range or ``NaN`` on
            failure, and ``raw_output`` contains the model response.
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
        result, exc = try_call_with_policy(
            messages,
            invocation=invocation,
            retry=policy,
            logger=LOGGER,
        )
        if result is None:
            return float("nan"), f"(error after {policy.attempts} attempts: {exc})"
        raw_output = result

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

        return clip_prediction(value), raw_output

    @staticmethod
    def _extract_log_messages(messages: Sequence[Mapping[str, str]]) -> Tuple[str, str]:
        """Return the system prompt and latest user message for QA logging.

        :param messages: Chat message sequence used for inference.
        :returns: Tuple of ``(system_prompt, user_prompt)`` strings (may be empty).
        :rtype: tuple[str, str]
        """
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
        """Return the artefact paths associated with ``spec``.

        :param spec: Opinion study under evaluation.
        :returns: Paths for metrics, predictions, and QA log files.
        :rtype: OpinionArtifacts
        """
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
        """Persist predictions to disk.

        :param artifacts: Destination artefact paths for the study.
        :param predictions: Serialised per-participant payloads.
        :returns: ``None``.
        """
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
        """Return cached study metrics when available and well-formed.

        :param metrics_path: Path to the saved ``metrics.json``.
        :param study_key: Study identifier used for logging.
        :param default_label: Fallback study label when missing in the payload.
        :param default_issue: Fallback issue label when missing in the payload.
        :returns: Parsed :class:`CachedStudyPayload` or ``None`` if unreadable.
        :rtype: CachedStudyPayload | None
        """
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
        """Return cached prediction vectors when available and well-formed.

        :param predictions_path: Path to the saved ``predictions.jsonl`` file.
        :param study_key: Study identifier used for logging.
        :returns: Parsed :class:`CachedPredictionVectors` or ``None`` if
            predictions cannot be read.
        :rtype: CachedPredictionVectors | None
        """
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
        """Write the metrics JSON payload for ``spec``.

        :param spec: Study specification providing labels for the payload.
        :param artifacts: Output paths for metrics/predictions files.
        :param payload: Aggregated metrics and participant counts.
        :returns: ``None``.
        """
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
        """Load cached artefacts when reuse is permitted.

        :param spec: Study specification identifying cached directories.
        :param artifacts: Expected artefact paths for the study.
        :param accumulator: Combined accumulator to extend with cached vectors.
        :returns: :class:`OpinionStudyResult` reconstructed from disk or ``None``.
        """
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
            metrics=metrics_payload.metrics,
            baseline=metrics_payload.baseline,
            artifacts=artifacts,
            spec=spec,
        )

    def _write_qa_log_entry(self, qa_log, entry: QALogEntry) -> None:
        """Write a single QA log entry for the participant.

        :param qa_log: Open file handle for the QA log.
        :param entry: :class:`QALogEntry` containing prompts and model output.
        :returns: ``None``.
        """
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
        """Run inference for a single participant example and update artefacts.

        :param spec: Study specification providing labels and issue.
        :param example: Canonical participant payload with ``before``/``after``.
        :param idx: 1-based participant index used in QA logs.
        :param context: Shared batch/logging resources.
        :returns: ``None``.
        """
        messages = self._build_messages(spec, example)
        prediction, raw_output = self._infer_prediction(messages)
        if math.isnan(prediction):
            prediction = example["before"]
        prediction = clip_prediction(prediction)

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
        """Evaluate ``spec`` and update the combined accumulator.

        :param spec: Opinion study specification to evaluate.
        :param dataset_rows: Materialised dataset rows for the split.
        :param accumulator: Combined accumulator updated with vectors.
        :returns: :class:`OpinionStudyResult` when examples exist, else ``None``.
        """
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
        baseline = baseline_metrics(batch.truth_before, batch.truth_after)

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
            metrics=metrics,
            baseline=baseline,
            artifacts=artifacts,
            spec=spec,
        )

    def run(self) -> OpinionEvaluationResult:
        """Execute the opinion evaluation for the configured studies.

        :returns: Aggregated :class:`OpinionEvaluationResult` including per-study
            results and combined metrics.
        :rtype: OpinionEvaluationResult
        """
        dataset_rows = self._load_dataset()
        specs_by_key: Dict[str, OpinionSpec] = {spec.key.lower(): spec for spec in DEFAULT_SPECS}
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
    """Execute GPT-4o opinion evaluations for the requested studies.

    :param args: CLI namespace or object used to build runtime/settings.
    :param config_label: Label describing the selected GPT-4o configuration.
    :param out_dir: Root directory for opinion artefacts.
    :returns: Final :class:`OpinionEvaluationResult` payload.
    :rtype: OpinionEvaluationResult
    """
    runner = OpinionEvaluationRunner(args, config_label=config_label, out_dir=out_dir)
    return runner.run()


__all__ = ["OpinionEvaluationRunner", "OpinionEvaluationResult", "run_opinion_evaluations"]

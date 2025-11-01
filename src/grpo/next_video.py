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

"""Next-video evaluation runner for finetuned GRPO checkpoints."""

from __future__ import annotations

import importlib as _importlib  # avoid triggering gpt4o.core lazy attribute sentinel
import json
import logging
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Sequence

from common.evaluation import slate_eval
from common.evaluation.prediction_rows import (
    parse_common_row_fields,
    observation_from_row as _obs_from_row_common,
)
from common.pipeline.io import (
    load_metrics_json,
    write_metrics_json,
    write_segmented_markdown_log,
    iter_jsonl_rows,
)
from common.chat.utils import latest_user_content
from .dataset import PreparedExample, load_dataset_split, prepare_examples
from .model import generate_chat_completion

gpt4o_utils = _importlib.import_module("gpt4o.core.utils")

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .model import GenerationSettings, ModelLike, TokenizerLike

LOGGER = logging.getLogger("grpo.next_video")

def _parse_index(raw_output: str) -> int | None:
    """Parse the predicted option index from raw model output.

    :param raw_output: Decoded model completion containing prediction tags.
    :returns: 1-based option index when a numeric answer is detected, else ``None``.
    :rtype: int | None
    """

    match = gpt4o_utils.ANS_TAG.search(raw_output)
    if match:
        candidate = match.group(1).strip()
        numeric = gpt4o_utils.INDEX_ONLY.match(candidate)
        if numeric:
            try:
                return int(numeric.group(1))
            except ValueError:
                return None
    tail = "\n".join(raw_output.strip().splitlines()[-4:])
    for line in reversed(tail.splitlines()):
        numeric = gpt4o_utils.INDEX_ONLY.match(line.strip())
        if numeric:
            try:
                return int(numeric.group(1))
            except ValueError:
                return None
    return None


@dataclass(frozen=True)
class FilterSelection:
    """Lower-cased filters applied before evaluation.

    :ivar tuple[str, ...] issues: Issue labels required for inclusion.
    :ivar tuple[str, ...] studies: Participant-study identifiers required for inclusion.
    """

    issues: tuple[str, ...]
    studies: tuple[str, ...]

    def allows(self, example: PreparedExample) -> bool:
        """Return ``True`` when the example passes the configured filters.

        :param example: Prepared example describing a single evaluation row.
        :returns: ``True`` if the example should be evaluated; otherwise ``False``.
        :rtype: bool
        """

        issue = example.issue.lower()
        study = example.participant_study.lower()
        if self.issues and issue not in self.issues:
            return False
        if self.studies and study not in self.studies:
            return False
        return True

    @classmethod
    def from_raw(
        cls,
        *,
        issues: Sequence[str] | None,
        studies: Sequence[str] | None,
    ) -> "FilterSelection":
        """Build a :class:`FilterSelection` from raw CLI tokens.

        :param issues: Raw issue filter tokens provided by the CLI.
        :param studies: Raw participant-study filter tokens.
        :returns: Normalised :class:`FilterSelection` instance.
        :rtype: FilterSelection
        """

        issue_tokens = tuple(sorted({token.strip().lower() for token in issues or () if token}))
        study_tokens = tuple(sorted({token.strip().lower() for token in studies or () if token}))
        return cls(issue_tokens, study_tokens)

    def metrics_filters(self) -> slate_eval.EvaluationFilters:
        """Return the :class:`EvaluationFilters` payload for metrics.

        :returns: Filters compatible with :mod:`common.evaluation.slate_eval`.
        :rtype: slate_eval.EvaluationFilters
        """

        return slate_eval.EvaluationFilters(issues=list(self.issues), studies=list(self.studies))


@dataclass(frozen=True)
class NextVideoDatasetSpec:
    """Dataset selection applied when loading evaluation rows.

    :ivar str name: Local path or Hugging Face dataset identifier.
    :ivar str split: Dataset split name used for evaluation.
    :ivar str | None cache_dir: Optional datasets cache directory.
    """

    name: str
    split: str
    cache_dir: str | None


@dataclass(frozen=True)
class NextVideoPromptSettings:
    """Prompt parameters used when preparing evaluation examples.

    :ivar str system_prompt: Base system prompt for the model.
    :ivar str | None solution_key: Dataset column containing the correct id.
    :ivar int max_history: Maximum number of history turns to include.
    """

    system_prompt: str
    solution_key: str | None
    max_history: int


@dataclass(frozen=True)
class NextVideoEvaluationLimits:
    """Caps and cadence controlling the evaluation runtime.

    :ivar int max_examples: Maximum number of examples to process; 0 disables the cap.
    :ivar int flush_every: Periodic flush interval in examples; 0 disables flushing.
    """

    max_examples: int
    flush_every: int = 0

    def example_cap(self) -> int | None:
        """Return the maximum number of examples to evaluate, if any.

        :returns: Positive integer limit or ``None`` when unrestricted.
        :rtype: int | None
        """

        return self.max_examples if self.max_examples > 0 else None


@dataclass(frozen=True)
class NextVideoEvaluationSettings:
    """Configuration bundle controlling next-video evaluation.

    :ivar str model_label: Human-readable label associated with the model.
    :ivar NextVideoDatasetSpec dataset: Dataset selection parameters.
    :ivar NextVideoPromptSettings prompts: Prompt configuration for examples.
    :ivar NextVideoEvaluationLimits limits: Runtime evaluation limits.
    :ivar bool overwrite: Whether to overwrite existing artefacts.
    :ivar GenerationSettings generation: Generation hyperparameters.
    :ivar FilterSelection filters: Pre-evaluation filter configuration.
    """

    model_label: str
    dataset: NextVideoDatasetSpec
    prompts: NextVideoPromptSettings
    limits: NextVideoEvaluationLimits
    overwrite: bool
    generation: 'GenerationSettings'
    filters: FilterSelection


@dataclass(frozen=True)
class NextVideoEvaluationResult:
    """Aggregated artefacts produced by the evaluation runner.

    :ivar Path run_dir: Evaluation run directory.
    :ivar Path metrics_path: Path to the metrics JSON file.
    :ivar Path predictions_path: Path to the predictions JSONL.
    :ivar Path qa_log_path: Path to the rendered QA log.
    :ivar Mapping[str, object] metrics: In-memory metrics payload.
    """

    run_dir: Path
    metrics_path: Path
    predictions_path: Path
    qa_log_path: Path
    metrics: Mapping[str, object]


@dataclass(frozen=True)
class _ExampleArtefacts:
    """Evaluation artefacts derived from a single prediction.

    :ivar slate_eval.Observation observation: Metrics observation payload.
    :ivar Mapping[str, object] prediction: JSONL-friendly prediction record.
    :ivar str qa_entry: Markdown block appended to the QA log.
    """

    observation: slate_eval.Observation
    prediction: Mapping[str, object]
    qa_entry: str


@dataclass
class _NextVideoRunState:
    """Mutable evaluation buffers shared across the next-video loop."""

    accumulator: slate_eval.EvaluationAccumulator
    example_cap: int | None
    predictions: List[Mapping[str, object]] = field(default_factory=list)
    qa_entries: List[str] = field(default_factory=list)

    def can_accept_more(self) -> bool:
        """Return ``True`` while the example budget allows another prediction."""

        return self.example_cap is None or len(self.predictions) < self.example_cap

    def record(self, artefacts: _ExampleArtefacts) -> None:
        """
        Append artefacts derived from a single evaluated example.

        :param artefacts: Bundle of observation, serialized prediction row, and
            QA log entry created for one evaluated example.
        :returns: ``None``.
        """

        self.accumulator.observe(artefacts.observation)
        self.predictions.append(artefacts.prediction)
        self.qa_entries.append(artefacts.qa_entry)


def _artefacts_from_saved_row(row: Mapping[str, object]) -> _ExampleArtefacts:
    """Reconstruct evaluation artefacts from a saved prediction row."""

    (
        issue,
        study,
        _,
        _,
        gold_index,
        parsed_index,
        _,
        _,
    ) = parse_common_row_fields(row)
    messages = row.get("messages")
    raw_output = str(row.get("gpt_output") or "")
    is_formatted = bool(gpt4o_utils.ANS_TAG.search(raw_output)) if raw_output else False

    observation = _obs_from_row_common(row, is_formatted=is_formatted)
    prediction = dict(row)
    qa_entry = "\n".join(
        [
            "## Example",
            f"- Issue: {issue}",
            f"- Participant study: {study}",
            "",
            "### Prompt",
            json.dumps(messages, indent=2, ensure_ascii=False) if messages is not None else "",
            "",
            "### Model output",
            raw_output.strip(),
            "",
            f"### Parsed index: {parsed_index}",
            f"### Gold index: {gold_index}",
        ]
    )
    return _ExampleArtefacts(observation=observation, prediction=prediction, qa_entry=qa_entry)


def _seed_state_from_predictions(path: Path) -> _NextVideoRunState:
    """Return run state pre-populated from existing predictions JSONL."""

    state = _NextVideoRunState(
        accumulator=slate_eval.EvaluationAccumulator(),
        example_cap=None,
    )
    if not path.exists():
        return state
    for row in iter_jsonl_rows(path):
        artefacts = _artefacts_from_saved_row(row)
        state.record(artefacts)
    return state


@dataclass(frozen=True)
class _NextVideoOutputPaths:
    """Filesystem destinations for next-video evaluation artefacts."""

    predictions: Path
    metrics: Path
    qa_log: Path


@dataclass(frozen=True)
class _ResumeState:
    """Cached progress recovered from a previous run, if any."""

    seeded_state: _NextVideoRunState | None
    processed_count: int
    resume: bool


def _load_cached_result(run_dir: Path) -> NextVideoEvaluationResult | None:
    """Return a ready result if ``metrics.json`` already exists in ``run_dir``.

    This supports fast exits when a full evaluation has previously completed.
    """

    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    predictions_path = run_dir / "predictions.jsonl"
    qa_log = _qa_log_path(run_dir)
    cached_payload = load_metrics_json(metrics_path)
    metrics = cached_payload.get("metrics", cached_payload)
    return NextVideoEvaluationResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        qa_log_path=qa_log,
        metrics=metrics,
    )


def _resume_state(run_dir: Path) -> _ResumeState:
    """Return resume information derived from any existing predictions log."""

    predictions_path = run_dir / "predictions.jsonl"
    seeded_state = _seed_state_from_predictions(predictions_path)
    processed = len(seeded_state.predictions)
    return _ResumeState(seeded_state=seeded_state, processed_count=processed, resume=processed > 0)


def _build_metrics_request(settings: NextVideoEvaluationSettings) -> slate_eval.SlateMetricsRequest:
    """Construct the metrics request object from evaluation settings."""

    return slate_eval.SlateMetricsRequest(
        model_name=settings.model_label,
        dataset_name=settings.dataset.name,
        eval_split=settings.dataset.split,
        filters=settings.filters.metrics_filters(),
    )


def _build_example_artefacts(example: PreparedExample, response: str) -> _ExampleArtefacts:
    """Return the observation, prediction payload, and QA log entry for ``example``.

    :param example: Prepared example evaluated by the model.
    :param response: Raw model completion associated with the example.
    :returns: Structured artefacts used for metrics and reporting.
    :rtype: _ExampleArtefacts
    """

    parsed_index = _parse_index(response)
    option_bucket = slate_eval.bucket_from_options(example.n_options)
    position_bucket = slate_eval.bucket_from_position(example.position_index)
    eligible = example.gold_index > 0 and example.n_options > 0
    is_correct = eligible and parsed_index == example.gold_index
    is_formatted = bool(gpt4o_utils.ANS_TAG.search(response))

    observation = slate_eval.Observation(
        issue_label=example.issue or "unspecified",
        study_label=example.participant_study or "unspecified",
        position_bucket=position_bucket,
        option_bucket=option_bucket,
        option_count=example.n_options,
        gold_index=example.gold_index,
        parsed_index=parsed_index,
        is_formatted=is_formatted,
        eligible=eligible,
        is_correct=is_correct,
    )
    prediction = {
        "messages": example.messages,
        "gpt_output": response,
        "parsed_index": parsed_index,
        "gold_index": example.gold_index,
        "n_options": example.n_options,
        "correct": bool(is_correct),
        "eligible": bool(eligible),
        "issue": observation.issue_label,
        "participant_study": observation.study_label,
        "position_index": example.position_index,
        "position_bucket": position_bucket,
    }
    qa_entry = "\n".join(
        [
            "## Example",
            f"- Issue: {observation.issue_label}",
            f"- Participant study: {observation.study_label}",
            "",
            "### Prompt",
            json.dumps(example.messages, indent=2, ensure_ascii=False),
            "",
            "### Model output",
            response.strip(),
            "",
            f"### Parsed index: {parsed_index}",
            f"### Gold index: {example.gold_index}",
        ]
    )
    return _ExampleArtefacts(observation=observation, prediction=prediction, qa_entry=qa_entry)


def _evaluate_examples(
    examples: Iterable[PreparedExample],
    *,
    settings: NextVideoEvaluationSettings,
    model: 'ModelLike',
    tokenizer: 'TokenizerLike',
    example_cap: int | None,
    initial_state: _NextVideoRunState | None = None,
    start_from: int = 0,
    # Optional progress hook called after each processed example.
    on_progress: 'callable[[int, _NextVideoRunState], None] | None' = None,
) -> _NextVideoRunState:
    """Return aggregated evaluation state for the provided examples."""

    state = initial_state or _NextVideoRunState(
        accumulator=slate_eval.EvaluationAccumulator(),
        example_cap=example_cap,
    )

    for idx, example in enumerate(examples, start=1):
        if not state.can_accept_more():
            break
        if not settings.filters.allows(example):
            continue
        response = generate_chat_completion(
            model,
            tokenizer,
            example.messages,
            settings=settings.generation,
        )
        artefacts = _build_example_artefacts(example, response)
        state.record(artefacts)
        # Emit a per-example QA log line with correctness and running accuracy.
        question = latest_user_content(example.messages)
        running_acc = state.accumulator.accuracy()
        LOGGER.info(
            (
                "[NEXT][%d] issue=%s study=%s correct=%s parsed=%s gold=%s acc=%.3f\n"
                "Question: %s\n"
                "Answer: %s"
            ),
            start_from + idx,
            artefacts.prediction.get("issue", "unspecified"),
            artefacts.prediction.get("participant_study", "unspecified"),
            artefacts.prediction.get("correct", False),
            artefacts.prediction.get("parsed_index"),
            artefacts.prediction.get("gold_index"),
            running_acc,
            question,
            response.strip(),
        )
        # Also mirror a concise summary to stdout for SLURM logs.
        print(
            (
                f"[grpo.next_video] ex={start_from + idx} "
                f"issue={artefacts.prediction.get('issue', 'unspecified')} "
                f"study={artefacts.prediction.get('participant_study', 'unspecified')} "
                f"correct={str(artefacts.prediction.get('correct', False)).lower()} "
                f"parsed={artefacts.prediction.get('parsed_index')} "
                f"gold={artefacts.prediction.get('gold_index')} "
                f"acc={running_acc:.3f}"
            ),
            flush=True,
        )
        if idx % 25 == 0:
            LOGGER.info(
                "[NEXT] processed=%d accuracy=%.3f parsed=%.3f formatted=%.3f",
                idx,
                state.accumulator.accuracy(),
                state.accumulator.parsed_rate(),
                state.accumulator.format_rate(),
            )
        # Periodic flush callback (resilience). Called after each processed example
        # so the hook can decide to persist artefacts at a chosen cadence.
        if on_progress is not None:
            try:
                on_progress(start_from + idx, state)
            except (OSError, IOError, ValueError, TypeError):  # pragma: no cover - defensive
                LOGGER.warning("Progress hook raised; continuing without flush.")

    return state


def _ensure_output_dir(run_dir: Path, overwrite: bool) -> None:
    """Create or clear the run directory depending on ``overwrite``.

    :param run_dir: Target directory used for storing evaluation artefacts.
    :param overwrite: When ``False`` raise if the directory already exists.
    :returns: ``None``. The target directory is created on success.
    :raises FileExistsError: If ``run_dir`` exists and ``overwrite`` is ``False``.
    """

    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{run_dir}' already exists. Use --overwrite to refresh."
            )
    run_dir.mkdir(parents=True, exist_ok=True)


def _qa_log_path(run_dir: Path) -> Path:
    """Mirror the GPT-4o logging structure under ``logs/grpo``.

    :param run_dir: Evaluation run directory.
    :returns: Destination path for the QA log file.
    :rtype: Path
    """

    repo_logs = Path(__file__).resolve().parents[2] / "logs" / "grpo"
    try:
        relative = run_dir.relative_to(Path(__file__).resolve().parents[2] / "models" / "grpo")
    except ValueError:
        relative = Path(run_dir.name)
    destination = repo_logs / relative
    destination.mkdir(parents=True, exist_ok=True)
    return destination / "qa.log"


def _write_predictions(path: Path, predictions: Sequence[Mapping[str, object]]) -> None:
    """Persist prediction payloads as JSONL.

    :param path: Output file path for the JSONL artefact.
    :param predictions: Iterable of prediction payloads to serialise.
    :returns: ``None``. The file is written to disk.
    """

    with path.open("w", encoding="utf-8") as handle:
        for entry in predictions:
            handle.write(json.dumps(entry, ensure_ascii=False))
            handle.write("\n")


def run_next_video_evaluation(
    *,
    tokenizer: 'TokenizerLike',
    model: 'ModelLike',
    settings: NextVideoEvaluationSettings,
    config_label: str,
    out_dir: Path,
) -> NextVideoEvaluationResult:
    """Execute next-video evaluation for the provided GRPO checkpoint.

    :param tokenizer: Tokenizer configured for the GRPO checkpoint.
    :param model: Loaded causal language model under evaluation.
    :param settings: Evaluation configuration bundle.
    :param config_label: Label identifying the current configuration run.
    :param out_dir: Base directory receiving evaluation outputs.
    :returns: Next-video evaluation artefacts produced during execution.
    :rtype: NextVideoEvaluationResult
    """
    LOGGER.info(
        "[NEXT] loading dataset name=%s split=%s issues=%s studies=%s overwrite=%s",
        settings.dataset.name,
        settings.dataset.split,
        ",".join(settings.filters.issues) or "<any>",
        ",".join(settings.filters.studies) or "<any>",
        settings.overwrite,
    )
    print(
        (
            f"[grpo.next_video] dataset={settings.dataset.name} "
            f"split={settings.dataset.split} "
            f"issues={','.join(settings.filters.issues) or '<any>'} "
            f"studies={','.join(settings.filters.studies) or '<any>'} "
            f"overwrite={settings.overwrite}"
        ),
        flush=True,
    )

    rows = load_dataset_split(
        settings.dataset.name,
        split=settings.dataset.split,
        cache_dir=settings.dataset.cache_dir,
    )
    LOGGER.info("[NEXT] loaded %d raw rows for evaluation", len(rows))
    print(f"[grpo.next_video] loaded {len(rows)} raw rows", flush=True)
    examples_iter = prepare_examples(
        rows,
        system_prompt=settings.prompts.system_prompt,
        solution_key=settings.prompts.solution_key,
        max_history=settings.prompts.max_history,
    )
    run_dir = out_dir / config_label
    try:
        _ensure_output_dir(run_dir, settings.overwrite)
        resume_info = _ResumeState(seeded_state=None, processed_count=0, resume=False)
    except FileExistsError:
        cached = _load_cached_result(run_dir)
        if cached is not None:
            LOGGER.info("[NEXT] found cached artefacts at %s; skipping re-evaluation.", run_dir)
            return cached
        resume_info = _resume_state(run_dir)
        if resume_info.resume:
            LOGGER.info(
                "[NEXT] resuming evaluation at %d processed examples in %s",
                resume_info.processed_count,
                run_dir,
            )
    if settings.limits.example_cap() is not None:
        LOGGER.info("[NEXT] limiting evaluation to %d examples", settings.limits.example_cap())
        print(
            f"[grpo.next_video] limiting to {settings.limits.example_cap()} examples",
            flush=True,
        )
    LOGGER.info("[NEXT] writing artefacts to %s", run_dir)
    print(f"[grpo.next_video] writing artefacts to {run_dir}", flush=True)

    # Apply example cap relative to any already processed examples.
    cap = settings.limits.example_cap()
    if resume_info.processed_count and cap is not None:
        cap = max(0, cap - resume_info.processed_count)
    # Skip already processed examples if resuming.
    if resume_info.processed_count:
        examples_iter = islice(examples_iter, resume_info.processed_count, None)

    # Prepare outputs and request up-front for periodic flushing if enabled.
    outputs = _NextVideoOutputPaths(
        predictions=run_dir / "predictions.jsonl",
        metrics=run_dir / "metrics.json",
        qa_log=_qa_log_path(run_dir),
    )
    def _maybe_flush(processed_abs: int, state: _NextVideoRunState) -> None:
        flush_every = int(settings.limits.flush_every or 0)
        if flush_every <= 0:
            return
        if processed_abs % flush_every != 0:
            return
        # Rewrite predictions/metrics/QA log with current buffers. This supports
        # resume and enables report generation from partial runs.
        try:
            _write_predictions(outputs.predictions, state.predictions)
            metrics = state.accumulator.metrics_payload(_build_metrics_request(settings))
            write_metrics_json(outputs.metrics, metrics)
            write_segmented_markdown_log(
                outputs.qa_log,
                title="GRPO Next-Video QA Log",
                entries=state.qa_entries,
            )
            LOGGER.debug("[NEXT] flushed artefacts at %d examples", processed_abs)
        except (OSError, IOError, ValueError, TypeError):
            LOGGER.warning("[NEXT] flush at %d examples failed; continuing.", processed_abs)

    state = _evaluate_examples(
        examples_iter,
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        example_cap=cap,
        initial_state=resume_info.seeded_state,
        start_from=resume_info.processed_count,
        on_progress=_maybe_flush,
    )
    metrics = state.accumulator.metrics_payload(_build_metrics_request(settings))
    _write_predictions(outputs.predictions, state.predictions)
    write_metrics_json(outputs.metrics, metrics)
    write_segmented_markdown_log(
        outputs.qa_log,
        title="GRPO Next-Video QA Log",
        entries=state.qa_entries,
    )

    LOGGER.info(
        "[NEXT] complete accuracy=%.3f parsed=%.3f formatted=%.3f eligible=%d/%d",
        state.accumulator.accuracy(),
        state.accumulator.parsed_rate(),
        state.accumulator.format_rate(),
        state.accumulator.eligible_overall,
        state.accumulator.total_seen,
    )
    print(
        (
            f"[grpo.next_video] complete accuracy={state.accumulator.accuracy():.3f} "
            f"parsed={state.accumulator.parsed_rate():.3f} "
            f"formatted={state.accumulator.format_rate():.3f} "
            f"eligible={state.accumulator.eligible_overall}/{state.accumulator.total_seen}"
        ),
        flush=True,
    )
    return NextVideoEvaluationResult(
        run_dir=run_dir,
        metrics_path=outputs.metrics,
        predictions_path=outputs.predictions,
        qa_log_path=outputs.qa_log,
        metrics=metrics,
    )

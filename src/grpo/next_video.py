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

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Mapping, Sequence

from common.evaluation import slate_eval
from common.pipeline.io import write_metrics_json, write_segmented_markdown_log
from gpt4o import utils as gpt4o_utils

from .dataset import PreparedExample, load_dataset_split, prepare_examples
from .model import generate_chat_completion

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
    """Caps controlling the evaluation runtime.

    :ivar int max_examples: Maximum number of examples to process; 0 disables the cap.
    """

    max_examples: int

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
        """Append artefacts derived from a single evaluated example."""

        self.accumulator.observe(artefacts.observation)
        self.predictions.append(artefacts.prediction)
        self.qa_entries.append(artefacts.qa_entry)


@dataclass(frozen=True)
class _NextVideoOutputPaths:
    """Filesystem destinations for next-video evaluation artefacts."""

    predictions: Path
    metrics: Path
    qa_log: Path


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
) -> _NextVideoRunState:
    """Return aggregated evaluation state for the provided examples."""

    state = _NextVideoRunState(
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
        if idx % 25 == 0:
            LOGGER.info(
                "[NEXT] processed=%d accuracy=%.3f parsed=%.3f formatted=%.3f",
                idx,
                state.accumulator.accuracy(),
                state.accumulator.parsed_rate(),
                state.accumulator.format_rate(),
            )

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

    rows = load_dataset_split(
        settings.dataset.name,
        split=settings.dataset.split,
        cache_dir=settings.dataset.cache_dir,
    )
    examples_iter = prepare_examples(
        rows,
        system_prompt=settings.prompts.system_prompt,
        solution_key=settings.prompts.solution_key,
        max_history=settings.prompts.max_history,
    )
    run_dir = out_dir / config_label
    _ensure_output_dir(run_dir, settings.overwrite)

    state = _evaluate_examples(
        examples_iter,
        settings=settings,
        model=model,
        tokenizer=tokenizer,
        example_cap=settings.limits.example_cap(),
    )

    request = slate_eval.SlateMetricsRequest(
        model_name=settings.model_label,
        dataset_name=settings.dataset.name,
        eval_split=settings.dataset.split,
        filters=settings.filters.metrics_filters(),
    )
    metrics = state.accumulator.metrics_payload(request)

    outputs = _NextVideoOutputPaths(
        predictions=run_dir / "predictions.jsonl",
        metrics=run_dir / "metrics.json",
        qa_log=_qa_log_path(run_dir),
    )
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
    return NextVideoEvaluationResult(
        run_dir=run_dir,
        metrics_path=outputs.metrics,
        predictions_path=outputs.predictions,
        qa_log_path=outputs.qa_log,
        metrics=metrics,
    )

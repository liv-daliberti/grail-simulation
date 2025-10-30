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

# pylint: disable=too-many-branches,too-many-locals,too-many-statements
# pylint: disable=broad-exception-caught,duplicate-code

"""Evaluation routines for the GPT-4o slate-ranking baseline.

Fetches the configured dataset, issues batched GPT-4o requests, parses the
responses, and aggregates accuracy and formatting metrics for reporting.
"""

from __future__ import annotations

import json
import logging
import pathlib
import time
from dataclasses import dataclass
from importlib import import_module
import typing as t

import common.evaluation.utils as _eval_utils
import common.data.hf_datasets as _hf_datasets
from common.evaluation import slate_eval

_client = import_module("gpt4o.client")
_config = import_module("gpt4o.config")
_conversation = import_module("gpt4o.conversation")
_utils = import_module("gpt4o.utils")

if t.TYPE_CHECKING:  # pragma: no cover - typing only imports
    from argparse import Namespace
else:  # pragma: no cover - runtime fallback for type hints
    Namespace = object  # type: ignore[misc]

DOWNLOAD_CONFIG_CLS, LOAD_DATASET, LOAD_FROM_DISK = _hf_datasets.get_dataset_loaders()

LOGGER = logging.getLogger("gpt4o.evaluate")


def _parse_index_from_output(raw: str) -> int | None:
    """Parse the model's predicted index from raw completion text.

    :param raw: Completion text returned by the model.
    :returns: Parsed integer index (1-based) or ``None`` when absent.
    """
    match = _utils.ANS_TAG.search(raw)
    if match:
        candidate = match.group(1).strip()
        numeric = _utils.INDEX_ONLY.match(candidate)
        if numeric:
            try:
                return int(numeric.group(1))
            except Exception:
                return None
    tail = "\n".join(raw.strip().splitlines()[-4:])
    for line in reversed(tail.splitlines()):
        numeric = _utils.INDEX_ONLY.match(line.strip())
        if numeric:
            try:
                return int(numeric.group(1))
            except Exception:
                return None
    return None


def _ensure_output_dir(output_dir: pathlib.Path, overwrite: bool) -> None:
    """Create the evaluation output directory if it does not already exist.

    :param output_dir: Target directory for evaluation artifacts.
    :param overwrite: Whether pre-existing directories should be reused.
    :raises FileExistsError: When the directory exists and overwrite is disabled.
    """
    if output_dir.exists() and not overwrite:
        raise FileExistsError(
            f"Output directory '{output_dir}' already exists. Use --overwrite to replace."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


class ExampleLabels(t.NamedTuple):
    """Normalised issue/study labels used for filtering and reporting."""

    issue_label: str
    issue_key: str
    study_label: str
    study_key: str


class FilterSet(t.NamedTuple):
    """Container for requested filters and their lower-cased match set."""

    requested: list[str]
    active_keys: set[str]

    def allows(self, candidate: str) -> bool:
        """Return ``True`` when the candidate is not filtered out.

        :param candidate: Candidate key to check against the active filter set.
        :returns: ``True`` if the candidate is allowed; ``False`` otherwise.
        """
        return not self.active_keys or candidate in self.active_keys


class FilterState(t.NamedTuple):
    """Aggregate filters for issues and participant studies."""

    issues: FilterSet
    studies: FilterSet

    def allows(self, labels: ExampleLabels) -> bool:
        """Check whether an example passes the configured filters.

        :param labels: Issue and study labels associated with the example.
        :returns: ``True`` when the example should be evaluated.
        """
        return self.issues.allows(labels.issue_key) and self.studies.allows(
            labels.study_key
        )


class OutputPaths(t.NamedTuple):
    """Resolved filesystem locations for evaluation artifacts."""

    root: pathlib.Path
    predictions: pathlib.Path
    metrics: pathlib.Path
    qa_log: pathlib.Path

    @classmethod
    def build(cls, out_dir: str | pathlib.Path, overwrite: bool) -> "OutputPaths":
        """Validate and construct the output directory structure.

        :param out_dir: Target output directory (string or pathlike).
        :param overwrite: Whether existing directories may be reused.
        :returns: Populated ``OutputPaths`` instance.
        """
        root = pathlib.Path(out_dir)
        _ensure_output_dir(root, overwrite)
        qa_log_path = _utils.qa_log_path_for(root)
        return cls(
            root=root,
            predictions=root / "predictions.jsonl",
            metrics=root / "metrics.json",
            qa_log=qa_log_path,
        )


class EvaluationLimits(t.NamedTuple):
    """Encapsulates evaluation row limits for progress reporting."""

    eval_max: int
    target: int | None

    @classmethod
    def from_arg(cls, raw_eval_max: int) -> "EvaluationLimits":
        """Create limits from the CLI ``eval_max`` argument.

        :param raw_eval_max: Raw CLI ``eval_max`` value (0 keeps every row).
        :returns: Populated ``EvaluationLimits`` instance.
        """
        eval_max = int(raw_eval_max or 0)
        target = eval_max or None
        return cls(eval_max=eval_max, target=target)


@dataclass
class EvaluationState:
    """Mutable state tracked during evaluation."""

    split: str = _config.EVAL_SPLIT
    streaming: bool = False

    def set_split(self, split: str) -> None:
        """Update the active evaluation split.

        :param split: Dataset split identifier.
        :returns: ``None``.
        """

        self.split = split

    def set_streaming(self, streaming: bool) -> None:
        """Mark whether streaming mode is active.

        :param streaming: Flag indicating streaming mode usage.
        :returns: ``None``.
        """

        self.streaming = streaming


class EvaluationRunner:
    """Stateful helper that orchestrates GPT-4o evaluation."""

    def __init__(self, args: Namespace) -> None:
        """Initialise the runner with CLI arguments and derived configuration.

        :param args: Parsed command-line arguments produced by ``cli.build_parser``.
        """
        self.args = args
        self.dataset_name = str(getattr(args, "dataset", "") or _config.DATASET_NAME)
        logging.info("Loading dataset %s", self.dataset_name)

        self.output = OutputPaths.build(args.out_dir, args.overwrite)

        _eval_utils.ensure_hf_cache(args.cache_dir)

        issues_raw = str(getattr(args, "issues", "") or "")
        studies_raw = str(getattr(args, "studies", "") or "")
        requested_issues, issues_filter = self._parse_filter(issues_raw)
        requested_studies, studies_filter = self._parse_filter(studies_raw)
        self.filters = FilterState(
            issues=FilterSet(requested=requested_issues, active_keys=issues_filter),
            studies=FilterSet(requested=requested_studies, active_keys=studies_filter),
        )

        self.limits = EvaluationLimits.from_arg(getattr(args, "eval_max", 0))
        self.state = EvaluationState()

    @staticmethod
    def _parse_filter(raw: str) -> t.Tuple[list[str], set[str]]:
        """Normalise a comma-separated CLI filter.

        :param raw: Raw string supplied on the command line.
        :returns: Ordered tokens and the lowercase set used for filtering logic.
        :rtype: tuple[list[str], set[str]]
        """
        tokens: list[str] = []
        for segment in raw.split(","):
            candidate = segment.strip()
            if candidate:
                tokens.append(candidate)
        lowered = {token.lower() for token in tokens if token}
        if "all" in lowered:
            lowered.clear()
        return tokens, lowered

    def _load_dataset_iter(self) -> t.Iterable[dict[str, object]]:
        """Load the evaluation dataset, falling back to streaming when required.

        :returns: Iterable over dataset rows ready for evaluation.
        :rtype: collections.abc.Iterable[dict[str, object]]
        :raises RuntimeError: When neither local nor remote datasets can be loaded.
        """
        dataset_path = pathlib.Path(self.dataset_name)
        dataset = None

        if dataset_path.exists():
            _hf_datasets.require_dataset_support(needs_local=True)
            assert LOAD_FROM_DISK is not None  # narrow Optional for type checkers
            logging.info("Detected local dataset at %s", dataset_path)
            dataset = LOAD_FROM_DISK(str(dataset_path))  # type: ignore[arg-type]
        else:
            _hf_datasets.require_dataset_support()
            assert LOAD_DATASET is not None and DOWNLOAD_CONFIG_CLS is not None
            download_config = DOWNLOAD_CONFIG_CLS(
                resume_download=True,
                max_retries=2,
            )  # type: ignore[misc]
            try:
                dataset = LOAD_DATASET(
                    self.dataset_name,
                    cache_dir=self.args.cache_dir,
                    download_config=download_config,
                )
            except Exception as exc:
                message = str(exc)
                if "Not enough disk space" in message or "Insufficient space" in message:
                    logging.warning(
                        "Low disk space detected; falling back to streaming mode."
                    )
                    self.state.set_streaming(True)
                else:
                    raise

        assert LOAD_DATASET is not None

        if self.state.streaming:
            return self._load_streaming_split()
        return self._load_materialised_split(dataset)

    def _load_streaming_split(self) -> t.Iterable[dict[str, object]]:
        """Stream the evaluation split directly from Hugging Face datasets.

        :returns: Iterable delivering rows lazily without full materialisation.
        :rtype: collections.abc.Iterable[dict[str, object]]
        :raises RuntimeError: If the evaluation split cannot be streamed.
        """
        eval_split = _config.EVAL_SPLIT
        try:
            data_iter = LOAD_DATASET(  # type: ignore[misc]
                self.dataset_name,
                split=eval_split,
                streaming=True,
            )
        except Exception as exc:  # pragma: no cover - fallback path
            for fallback in ("validation", "eval", "test"):
                try:
                    data_iter = LOAD_DATASET(  # type: ignore[misc]
                        self.dataset_name,
                        split=fallback,
                        streaming=True,
                    )
                    eval_split = fallback
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError(
                    "Unable to load evaluation split in streaming mode."
                ) from exc

        self.state.set_split(eval_split)
        if self.limits.eval_max:
            data_iter = data_iter.take(self.limits.eval_max)
        return data_iter

    def _load_materialised_split(self, dataset: object) -> t.Iterable[dict[str, object]]:
        """Materialise the evaluation split from an in-memory dataset object.

        :param dataset: Dataset or DatasetDict object returned by ``datasets``.
        :returns: Iterable of rows suitable for evaluation.
        :rtype: collections.abc.Iterable[dict[str, object]]
        :raises RuntimeError: If ``dataset`` is missing when streaming is disabled.
        """
        if dataset is None:
            raise RuntimeError("Expected dataset object when not streaming.")

        eval_split = _config.EVAL_SPLIT
        available_splits: list[str] = []
        if hasattr(dataset, "keys"):
            try:
                available_splits = list(dataset.keys())  # type: ignore[assignment]
            except Exception:  # pragma: no cover - defensive, matches prior behaviour
                available_splits = []

        if available_splits:
            eval_split = next(
                (
                    split
                    for split in (_config.EVAL_SPLIT, "validation", "eval", "test")
                    if split in available_splits
                ),
                available_splits[0],
            )
            data_iter = dataset[eval_split]  # type: ignore[index]
        else:
            eval_split = (
                getattr(dataset, "split", None) or _config.EVAL_SPLIT
            )  # type: ignore[attr-defined]
            data_iter = dataset

        if self.limits.eval_max and hasattr(data_iter, "select"):
            limit = min(self.limits.eval_max, len(data_iter))  # type: ignore[arg-type]
            data_iter = data_iter.select(range(limit))

        self.state.set_split(eval_split)
        return data_iter

    @staticmethod
    def _prepare_labels(example: dict[str, object]) -> ExampleLabels:
        """Extract normalised issue and study labels from an example row.

        :param example: Dataset row containing issue and participant metadata.
        :returns: Lowercased and human-readable label bundle.
        :rtype: ExampleLabels
        """
        issue_raw = str(example.get("issue", "") or "").strip()
        issue_label = issue_raw if issue_raw else "unspecified"
        study_raw = str(example.get("participant_study", "") or "").strip()
        study_label = study_raw if study_raw else "unspecified"
        return ExampleLabels(
            issue_label=issue_label,
            issue_key=issue_label.lower(),
            study_label=study_label,
            study_key=study_label.lower(),
        )

    def _passes_filters(self, labels: ExampleLabels) -> bool:
        """Return ``True`` when the example satisfies issue and study filters.

        :param labels: Label bundle produced by ``_prepare_labels``.
        :returns: Boolean indicating whether the example should be evaluated.
        :rtype: bool
        """
        return self.filters.allows(labels)

    def _invoke_model(self, messages: list[dict[str, object]]) -> str:
        """Execute a GPT-4o call with retry semantics.

        :param messages: Chat completion payload produced for the example.
        :returns: Raw model output, or an error stub when retries fail.
        :rtype: str
        """
        retries = max(1, int(getattr(self.args, "request_retries", 5) or 5))
        delay = max(0.0, float(getattr(self.args, "request_retry_delay", 1.0) or 0.0))
        invocation = _utils.InvocationParams(
            max_tokens=self.args.max_tokens,
            temperature=self.args.temperature,
            top_p=getattr(self.args, "top_p", None),
            deployment=getattr(self.args, "deployment", None),
        )
        policy = _utils.RetryPolicy(attempts=retries, delay=delay)
        try:
            return _utils.call_gpt4o_with_retries(
                messages,
                invocation=invocation,
                retry=policy,
                logger=LOGGER,
            )
        except Exception as exc:  # pragma: no cover - best effort logging
            LOGGER.error(
                "GPT-4o call failed after %d attempts; returning error stub: %s",
                policy.attempts,
                exc,
            )
            return f"(error after {policy.attempts} attempts: {exc})"

    def _evaluate_example(
        self, example: dict[str, object]
    ) -> t.Tuple[slate_eval.Observation, dict[str, object]] | None:
        """Evaluate a single example and return observation metrics plus payloads.

        :param example: Dataset row containing slate information.
        :returns: Observation/payload tuple, or ``None`` when filtered.
        :rtype: tuple[slate_eval.Observation, dict[str, object]] | None
        """
        labels = self._prepare_labels(example)
        if not self._passes_filters(labels):
            return None

        record = _conversation.make_conversation_record(example)
        messages = record["prompt"]
        gold_index = int(record.get("gold_index", -1))
        option_count = int(record.get("n_options", 0))
        position_index = int(record.get("position_index", -1))

        raw_output = self._invoke_model(messages)
        is_formatted = bool(_utils.ANS_TAG.search(raw_output))
        parsed_index = _parse_index_from_output(raw_output)
        pos_bucket = slate_eval.bucket_from_position(position_index)
        option_bucket = slate_eval.bucket_from_options(option_count)
        eligible = gold_index > 0 and option_count > 0
        is_correct = (
            eligible and (parsed_index is not None) and (parsed_index == gold_index)
        )

        observation_fields = {
            "issue_label": labels.issue_label,
            "study_label": labels.study_label,
            "position_bucket": pos_bucket,
            "option_bucket": option_bucket,
            "option_count": option_count,
            "gold_index": gold_index,
            "parsed_index": parsed_index,
            "is_formatted": is_formatted,
            "eligible": eligible,
            "is_correct": is_correct,
        }
        observation = slate_eval.Observation(**observation_fields)
        payload = {
            "messages": messages,
            "gpt_output": raw_output,
            "parsed_index": parsed_index,
            "gold_index": gold_index,
            "n_options": option_count,
            "correct": bool(is_correct),
            "eligible": bool(eligible),
            "issue": labels.issue_label,
            "participant_study": labels.study_label,
            "position_index": position_index,
            "position_bucket": pos_bucket,
        }
        return observation, payload

    def _maybe_log_progress(
        self,
        seen_rows: int,
        start_time: float,
        accumulator: slate_eval.EvaluationAccumulator,
    ) -> None:
        """Emit periodic progress logs during evaluation.

        :param seen_rows: Number of rows processed so far.
        :param start_time: Epoch timestamp marking the evaluation start.
        :param accumulator: Aggregator containing current metric totals.
        """
        if seen_rows == 0 or seen_rows % 25:
            return
        elapsed = time.time() - start_time
        denom = self.limits.target if self.limits.target is not None else seen_rows
        logging.info(
            "[eval] %d/%s  acc=%.3f  parsed=%.3f  fmt=%.3f  %.1fs",
            seen_rows,
            denom,
            accumulator.accuracy(),
            accumulator.parsed_rate(),
            accumulator.format_rate(),
            elapsed,
        )

    def metrics_request(self) -> slate_eval.SlateMetricsRequest:
        """Construct the payload used when serialising evaluation metrics.

        :returns: Metrics request capturing model, dataset, split, and filters.
        """
        model_name = getattr(self.args, "deployment", None) or _config.DEPLOYMENT_NAME
        return slate_eval.SlateMetricsRequest(
            model_name=model_name,
            dataset_name=self.dataset_name,
            eval_split=self.state.split,
            filters=slate_eval.EvaluationFilters(
                issues=self.filters.issues.requested,
                studies=self.filters.studies.requested,
            ),
        )

    def _print_summary(self, accumulator: slate_eval.EvaluationAccumulator) -> None:
        """Print a concise evaluation summary and output file paths.

        :param accumulator: Aggregated evaluation metrics.
        :returns: ``None``.
        """
        summary_bits = [
            f"[DONE] dataset={self.dataset_name}",
            f"split={self.state.split}",
            f"n={accumulator.total_seen}",
            f"eligible={accumulator.eligible_overall}",
            f"accuracy={accumulator.accuracy():.4f}",
            f"parsed_ok={accumulator.parsed_rate():.3f}",
            f"format_rate={accumulator.format_rate():.3f}",
        ]
        summary = "  ".join(summary_bits)
        print(summary)
        print(f"[WROTE] per-example: {self.output.predictions}")
        print(f"[WROTE] metrics:     {self.output.metrics}")
        print(f"[WROTE] QA log:      {self.output.qa_log}")

    def run(self) -> None:
        """Execute the full evaluation workflow.

        :returns: ``None``.
        """
        data_iter = self._load_dataset_iter()
        accumulator = slate_eval.EvaluationAccumulator()
        start_time = time.time()
        seen_rows = 0

        with open(self.output.predictions, "w", encoding="utf-8") as writer, open(
            self.output.qa_log, "w", encoding="utf-8"
        ) as qa_log:
            for example in data_iter:
                result = self._evaluate_example(example)
                if result is None:
                    continue
                observation, payload = result
                seen_rows += 1
                accumulator.observe(observation)
                writer.write(json.dumps(payload, ensure_ascii=False) + "\n")

                question = ""
                for message in reversed(payload.get("messages", [])):
                    if isinstance(message, dict) and message.get("role") == "user":
                        question = str(message.get("content", "")).strip()
                        break
                answer = str(payload.get("gpt_output", "")).strip()
                issue_label = payload.get("issue")
                study_label = payload.get("participant_study")
                qa_log.write(f"### Example {seen_rows}\n")
                qa_log.write(f"Issue: {issue_label} | Study: {study_label}\n")
                system_prompt = ""
                for message in payload.get("messages", []):
                    if (
                        isinstance(message, dict)
                        and message.get("role") == "system"
                        and message.get("content")
                    ):
                        system_prompt = str(message["content"]).strip()
                        break
                qa_log.write("SYSTEM:\n")
                qa_log.write(f"{system_prompt}\n")
                qa_log.write("QUESTION:\n")
                qa_log.write(f"{question}\n")
                qa_log.write("ANSWER:\n")
                qa_log.write(f"{answer}\n")
                parsed_index = payload.get("parsed_index")
                gold_index = payload.get("gold_index")
                eligible = bool(payload.get("eligible"))
                correct = bool(payload.get("correct"))
                result_status = "ineligible"
                if eligible:
                    result_status = "correct" if correct else "wrong"
                pred_display = str(parsed_index) if parsed_index is not None else "None"
                gold_display = str(gold_index) if gold_index is not None else "None"
                qa_log.write(
                    f"RESULT: {result_status} (pred={pred_display}, gold={gold_display})\n\n"
                )
                qa_log.flush()

                self._maybe_log_progress(seen_rows, start_time, accumulator)

        metrics = accumulator.metrics_payload(self.metrics_request())
        with open(self.output.metrics, "w", encoding="utf-8") as handle:
            serialised = json.dumps(metrics, ensure_ascii=False, indent=2)
            handle.write(serialised)

        if accumulator.eligible_overall == 0:
            logging.warning(
                "Completed evaluation with zero eligible examples. "
                "Verify that the dataset includes gold annotations and the correct split."
            )

        self._print_summary(accumulator)


def run_eval(args: Namespace) -> None:
    """
    Evaluate GPT-4o on the configured dataset.

    :param args: Namespace with CLI parameters (temperature, max_tokens, eval_max, etc.)
    :type args: Namespace
    """

    EvaluationRunner(args).run()

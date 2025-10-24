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

# pylint: disable=duplicate-code

"""Evaluation routines for the GPT-4o slate-ranking baseline.

Fetches the configured dataset, issues batched GPT-4o requests, parses the
responses, and aggregates accuracy and formatting metrics for reporting.
"""

# pylint: disable=too-many-branches,too-many-locals,too-many-statements,broad-exception-caught,duplicate-code

from __future__ import annotations

import json
import logging
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from datasets import DownloadConfig, load_dataset, load_from_disk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DownloadConfig = None  # type: ignore
    load_dataset = None  # type: ignore
    load_from_disk = None  # type: ignore

from common.eval_utils import safe_div

from .client import ds_call
from .config import DATASET_NAME, DEPLOYMENT_NAME, EVAL_SPLIT
from .conversation import make_conversation_record
from .utils import ANS_TAG, INDEX_ONLY


@dataclass
class BucketAccumulator:
    """Accumulate per-bucket evaluation counts."""

    seen: Counter[str] = field(default_factory=Counter)
    eligible: Counter[str] = field(default_factory=Counter)
    correct: Counter[str] = field(default_factory=Counter)
    parsed: Counter[str] = field(default_factory=Counter)
    formatted: Counter[str] = field(default_factory=Counter)

    def record_seen(self, key: str) -> None:
        self.seen[key] += 1

    def record_formatted(self, key: str) -> None:
        self.formatted[key] += 1

    def record_parsed(self, key: str) -> None:
        self.parsed[key] += 1

    def record_eligible(self, key: str) -> None:
        self.eligible[key] += 1

    def record_correct(self, key: str) -> None:
        self.correct[key] += 1

    def summary(
        self,
        order: Sequence[str] | None = None,
        skip_empty: bool = False,
    ) -> Dict[str, Dict[str, float | int]]:
        """Return aggregated metrics per bucket."""

        keys = order if order is not None else sorted(self.seen.keys())
        result: Dict[str, Dict[str, float | int]] = {}
        for key in keys:
            seen = int(self.seen.get(key, 0))
            if skip_empty and seen == 0:
                continue
            eligible = int(self.eligible.get(key, 0))
            correct = int(self.correct.get(key, 0))
            parsed = int(self.parsed.get(key, 0))
            formatted = int(self.formatted.get(key, 0))
            result[key] = {
                "n_seen": seen,
                "n_eligible": eligible,
                "correct": correct,
                "accuracy": safe_div(correct, eligible),
                "parsed_rate": safe_div(parsed, seen),
                "format_rate": safe_div(formatted, seen),
            }
        return result

    def histogram(self, order: Sequence[str], attr: str) -> Dict[str, int]:
        """Return integer histogram for the requested attribute."""

        counter: Counter[str] = getattr(self, attr)
        return {key: int(counter.get(key, 0)) for key in order}

    def ratio(self, numerator: str, denominator: str, order: Sequence[str]) -> Dict[str, float]:
        """Return ratios for ``numerator`` divided by ``denominator``."""

        num: Counter[str] = getattr(self, numerator)
        denom: Counter[str] = getattr(self, denominator)
        return {
            key: safe_div(num.get(key, 0), denom.get(key, 0))
            for key in order
        }


@dataclass(frozen=True)
class Observation:
    """Snapshot of a single evaluation example."""

    issue_label: str
    study_label: str
    position_bucket: str
    option_bucket: str
    option_count: int
    gold_index: int
    parsed_index: Optional[int]
    is_formatted: bool
    eligible: bool
    is_correct: bool


@dataclass
class EvaluationAccumulator:
    """Stateful accumulator tracking evaluation metrics across examples."""

    position: BucketAccumulator = field(default_factory=BucketAccumulator)
    options: BucketAccumulator = field(default_factory=BucketAccumulator)
    issue: BucketAccumulator = field(default_factory=BucketAccumulator)
    study: BucketAccumulator = field(default_factory=BucketAccumulator)
    total_seen: int = 0
    format_ok: int = 0
    parsed_ok: int = 0
    eligible_overall: int = 0
    correct_overall: int = 0
    seen_single: int = 0
    seen_multi: int = 0
    eligible_single: int = 0
    eligible_multi: int = 0
    correct_single: int = 0
    correct_multi: int = 0
    parsed_multi: int = 0
    formatted_multi: int = 0
    gold_hist: Counter[int] = field(default_factory=Counter)
    all_gold_indices: list[int] = field(default_factory=list)
    option_counts: list[int] = field(default_factory=list)

    def observe(self, obs: Observation) -> None:
        """Update aggregates for a single example observation."""

        self.total_seen += 1
        self.position.record_seen(obs.position_bucket)
        self.issue.record_seen(obs.issue_label)
        self.study.record_seen(obs.study_label)
        self.options.record_seen(obs.option_bucket)

        if obs.is_formatted:
            self.format_ok += 1
            self.issue.record_formatted(obs.issue_label)
            self.study.record_formatted(obs.study_label)
            self.options.record_formatted(obs.option_bucket)
        if obs.parsed_index is not None:
            self.parsed_ok += 1
            self.issue.record_parsed(obs.issue_label)
            self.study.record_parsed(obs.study_label)
            self.options.record_parsed(obs.option_bucket)

        if obs.option_count == 1:
            self.seen_single += 1
        else:
            self.seen_multi += 1
            if obs.is_formatted:
                self.formatted_multi += 1
            if obs.parsed_index is not None:
                self.parsed_multi += 1

        if obs.eligible:
            self.eligible_overall += 1
            self.position.record_eligible(obs.position_bucket)
            self.issue.record_eligible(obs.issue_label)
            self.study.record_eligible(obs.study_label)
            self.options.record_eligible(obs.option_bucket)
            if obs.option_count == 1:
                self.eligible_single += 1
            else:
                self.eligible_multi += 1
            if obs.gold_index > 0:
                self.gold_hist[obs.gold_index] += 1
                self.all_gold_indices.append(obs.gold_index)
            if obs.option_count > 0:
                self.option_counts.append(obs.option_count)

        if obs.is_correct:
            self.correct_overall += 1
            self.position.record_correct(obs.position_bucket)
            self.issue.record_correct(obs.issue_label)
            self.study.record_correct(obs.study_label)
            self.options.record_correct(obs.option_bucket)
            if obs.option_count == 1:
                self.correct_single += 1
            else:
                self.correct_multi += 1

    def accuracy(self) -> float:
        """Return overall accuracy computed over eligible examples."""

        return safe_div(self.correct_overall, self.eligible_overall)

    def parsed_rate(self) -> float:
        """Return rate of successfully parsed completions."""

        return safe_div(self.parsed_ok, self.total_seen)

    def format_rate(self) -> float:
        """Return rate of outputs that matched the expected format."""

        return safe_div(self.format_ok, self.total_seen)

    def position_summary(self, order: Sequence[str]) -> Dict[str, Dict[str, float | int]]:
        """Return aggregated position metrics."""

        return self.position.summary(order=order)

    def options_summary(self, order: Sequence[str]) -> Dict[str, Any]:
        """Return aggregated option-count metrics."""

        return {
            "hist_seen": self.options.histogram(order, "seen"),
            "hist_eligible": self.options.histogram(order, "eligible"),
            "hist_correct": self.options.histogram(order, "correct"),
            "accuracy": self.options.ratio("correct", "eligible", order),
            "parsed_rate": self.options.ratio("parsed", "seen", order),
            "format_rate": self.options.ratio("formatted", "seen", order),
        }

    def group_summary(self) -> Dict[str, Dict[str, Dict[str, float | int]]]:
        """Return per-issue and per-study aggregates."""

        return {
            "by_issue": self.issue.summary(skip_empty=True),
            "by_participant_study": self.study.summary(skip_empty=True),
        }

    def single_multi_summary(self) -> Dict[str, float | int]:
        """Return metrics comparing single-option and multi-option prompts."""

        return {
            "n_single": int(self.seen_single),
            "n_multi": int(self.seen_multi),
            "eligible_single": int(self.eligible_single),
            "eligible_multi": int(self.eligible_multi),
            "accuracy_single": safe_div(self.correct_single, self.eligible_single),
            "accuracy_multi": safe_div(self.correct_multi, self.eligible_multi),
            "parsed_rate_multi": safe_div(self.parsed_multi, max(1, self.seen_multi)),
            "format_rate_multi": safe_div(self.formatted_multi, max(1, self.seen_multi)),
        }

    def baseline_metrics(self) -> tuple[Dict[str, int], Dict[str, Any], float]:
        """Return gold index distribution and baseline accuracies."""

        distribution = {str(key): int(value) for key, value in sorted(self.gold_hist.items())}
        if not distribution:
            baseline = {"top_index": None, "count": 0, "accuracy": 0.0}
            expected_random = 0.0
        else:
            top_index = max(self.gold_hist.items(), key=lambda kv: kv[1])[0]
            baseline = {
                "top_index": top_index,
                "count": int(self.gold_hist[top_index]),
                "accuracy": safe_div(
                    sum(1 for idx in self.all_gold_indices if idx == top_index),
                    self.eligible_overall,
                ),
            }
            expected_random = (
                float(np.mean([1.0 / count for count in self.option_counts]))
                if self.option_counts
                else 0.0
            )
        return distribution, baseline, expected_random

    def metrics_payload(
        self,
        model_name: str,
        dataset_name: str,
        eval_split: str,
        requested_issues: Sequence[str],
        requested_studies: Sequence[str],
    ) -> Dict[str, Any]:
        """Return the metrics blob written to disk after evaluation."""

        position_order = ["1", "2", "3", "4", "5+", "unknown"]
        option_order = ["1", "2", "3", "4", "5+"]
        gold_distribution, baseline_most_frequent, random_baseline = self.baseline_metrics()
        return {
            "model": model_name,
            "dataset": dataset_name,
            "split": eval_split,
            "n_total": int(self.total_seen),
            "n_eligible": int(self.eligible_overall),
            "accuracy_overall": self.accuracy(),
            "parsed_rate": self.parsed_rate(),
            "format_rate": self.format_rate(),
            "position_stats": self.position_summary(position_order),
            "by_n_options": self.options_summary(option_order),
            "split_single_vs_multi": self.single_multi_summary(),
            "group_metrics": self.group_summary(),
            "filters": {
                "issues": list(requested_issues),
                "studies": list(requested_studies),
            },
            "gold_index_distribution": gold_distribution,
            "baseline_most_frequent_gold_index": baseline_most_frequent,
            "random_baseline_expected_accuracy": random_baseline,
            "notes": (
                "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown."
            ),
        }


def _bucket_from_position(position_index: int) -> str:
    """Map a zero-based position index to an accuracy bucket label.

    :param position_index: Predicted slate position (zero-based).
    :returns: String bucket identifier used for metrics aggregation.
    """
    if position_index < 0:
        return "unknown"
    if position_index == 0:
        return "1"
    if position_index == 1:
        return "2"
    if position_index == 2:
        return "3"
    if position_index == 3:
        return "4"
    return "5+"


def _bucket_from_options(count: int) -> str:
    """Normalise the number of slate options into histogram buckets.

    :param count: Number of candidate options exposed to the model.
    :returns: String bucket identifier used for metrics aggregation.
    """
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count == 3:
        return "3"
    if count == 4:
        return "4"
    return "5+"


def _parse_index_from_output(raw: str) -> Optional[int]:
    """Parse the model's predicted index from raw completion text.

    :param raw: Completion text returned by the model.
    :returns: Parsed integer index (1-based) or ``None`` when absent.
    """
    match = ANS_TAG.search(raw)
    if match:
        candidate = match.group(1).strip()
        numeric = INDEX_ONLY.match(candidate)
        if numeric:
            try:
                return int(numeric.group(1))
            except Exception:
                return None
    tail = "\n".join(raw.strip().splitlines()[-4:])
    for line in reversed(tail.splitlines()):
        numeric = INDEX_ONLY.match(line.strip())
        if numeric:
            try:
                return int(numeric.group(1))
            except Exception:
                return None
    return None


def _ensure_output_dir(output_dir: Path, overwrite: bool) -> None:
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


def run_eval(args: Any) -> None:
    """
    Evaluate GPT-4o on the configured dataset.

    :param args: Namespace with CLI parameters (temperature, max_tokens, eval_max, etc.)
    :type args: Any
    """

    dataset_name = str(getattr(args, "dataset", "") or DATASET_NAME)
    logging.info("Loading dataset %s", dataset_name)
    out_dir = Path(args.out_dir)
    _ensure_output_dir(out_dir, args.overwrite)
    out_jsonl = out_dir / "predictions.jsonl"
    metrics_json = out_dir / "metrics.json"

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    dataset_path = Path(dataset_name)
    if dataset_path.exists() and load_from_disk is None:
        raise ImportError(
            "The 'datasets' package with load_from_disk support is required to load local datasets."
        )
    if not dataset_path.exists() and (load_dataset is None or DownloadConfig is None):
        raise ImportError(
            "The 'datasets' package is required to run GPT-4o evaluations. "
            "Install it with `pip install datasets`."
        )

    use_streaming = False
    dataset = None
    if dataset_path.exists():
        logging.info("Detected local dataset at %s", dataset_path)
        dataset = load_from_disk(str(dataset_path))
    else:
        download_config = DownloadConfig(resume_download=True, max_retries=2)
        try:
            dataset = load_dataset(
                dataset_name,
                cache_dir=args.cache_dir,
                download_config=download_config,
            )
        except Exception as exc:
            message = str(exc)
            if "Not enough disk space" in message or "Insufficient space" in message:
                logging.warning("Low disk space detected; falling back to streaming mode.")
                use_streaming = True
            else:
                raise

    if use_streaming:
        eval_split = EVAL_SPLIT
        try:
            data_iter = load_dataset(dataset_name, split=eval_split, streaming=True)
        except Exception as exc:
            for fallback in ("validation", "eval", "test"):
                try:
                    data_iter = load_dataset(dataset_name, split=fallback, streaming=True)
                    eval_split = fallback
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("Unable to load evaluation split in streaming mode.") from exc
    else:
        eval_split = EVAL_SPLIT
        available_splits: list[str] = []
        if hasattr(dataset, "keys"):
            try:
                available_splits = list(dataset.keys())  # type: ignore[assignment]
            except Exception:
                available_splits = []
        if available_splits:
            eval_split = next(
                (
                    split
                    for split in (EVAL_SPLIT, "validation", "eval", "test")
                    if split in available_splits
                ),
                available_splits[0],
            )
            data_iter = dataset[eval_split]  # type: ignore[index]
        else:
            eval_split = getattr(dataset, "split", None) or EVAL_SPLIT  # type: ignore[attr-defined]
            data_iter = dataset
        if args.eval_max and hasattr(data_iter, "select"):
            data_iter = data_iter.select(range(min(args.eval_max, len(data_iter))))

    if use_streaming and args.eval_max:
        data_iter = data_iter.take(args.eval_max)

    raw_issues = str(getattr(args, "issues", "") or "")
    requested_issues = [token.strip() for token in raw_issues.split(",") if token.strip()]
    issues_filter = {token.lower() for token in requested_issues if token}
    if "all" in issues_filter:
        issues_filter.clear()

    raw_studies = str(getattr(args, "studies", "") or "")
    requested_studies = [token.strip() for token in raw_studies.split(",") if token.strip()]
    studies_filter = {token.lower() for token in requested_studies if token}
    if "all" in studies_filter:
        studies_filter.clear()

    n_eval_target = args.eval_max if args.eval_max else None
    accumulator = EvaluationAccumulator()
    start_time = time.time()
    seen_rows = 0

    with open(out_jsonl, "w", encoding="utf-8") as writer:
        for example in data_iter:
            issue_raw = str(example.get("issue", "") or "").strip()
            issue_label = issue_raw if issue_raw else "unspecified"
            issue_key = issue_label.lower()
            if issues_filter and issue_key not in issues_filter:
                continue

            study_raw = str(example.get("participant_study", "") or "").strip()
            study_label = study_raw if study_raw else "unspecified"
            study_key = study_label.lower()
            if studies_filter and study_key not in studies_filter:
                continue

            seen_rows += 1
            record = make_conversation_record(example)
            messages = record["prompt"]
            gold_index = int(record.get("gold_index", -1))
            option_count = int(record.get("n_options", 0))
            position_index = int(record.get("position_index", -1))

            pos_bucket = _bucket_from_position(position_index)

            try:
                raw_output = ds_call(
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    deployment=getattr(args, "deployment", None),
                )
            except Exception as exc:  # pragma: no cover - best effort logging
                raw_output = f"(error: {exc})"

            is_formatted = bool(ANS_TAG.search(raw_output))
            parsed_index = _parse_index_from_output(raw_output)
            option_bucket = _bucket_from_options(option_count)

            eligible = gold_index > 0 and option_count > 0
            is_correct = (
                eligible and (parsed_index is not None) and (parsed_index == gold_index)
            )

            accumulator.observe(
                Observation(
                    issue_label=issue_label,
                    study_label=study_label,
                    position_bucket=pos_bucket,
                    option_bucket=option_bucket,
                    option_count=option_count,
                    gold_index=gold_index,
                    parsed_index=parsed_index,
                    is_formatted=is_formatted,
                    eligible=eligible,
                    is_correct=is_correct,
                )
            )

            writer.write(
                json.dumps(
                    {
                        "messages": messages,
                        "gpt_output": raw_output,
                        "parsed_index": parsed_index,
                        "gold_index": gold_index,
                        "n_options": option_count,
                        "correct": bool(is_correct),
                        "eligible": bool(eligible),
                        "issue": issue_label,
                        "participant_study": study_label,
                        "position_index": position_index,
                        "position_bucket": pos_bucket,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if seen_rows % 25 == 0:
                elapsed = time.time() - start_time
                denom = n_eval_target if n_eval_target is not None else seen_rows
                logging.info(
                    "[eval] %d/%s  acc=%.3f  parsed=%.3f  fmt=%.3f  %.1fs",
                    seen_rows,
                    denom,
                    accumulator.accuracy(),
                    accumulator.parsed_rate(),
                    accumulator.format_rate(),
                    elapsed,
                )

    metrics = accumulator.metrics_payload(
        getattr(args, "deployment", None) or DEPLOYMENT_NAME,
        dataset_name,
        eval_split,
        requested_issues,
        requested_studies,
    )

    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    summary_bits = [
        f"[DONE] dataset={dataset_name}",
        f"split={eval_split}",
        f"n={accumulator.total_seen}",
        f"eligible={accumulator.eligible_overall}",
        f"accuracy={accumulator.accuracy():.4f}",
        f"parsed_ok={accumulator.parsed_rate():.3f}",
        f"format_rate={accumulator.format_rate():.3f}",
    ]
    summary = "  ".join(summary_bits)
    print(summary)
    print(f"[WROTE] per-example: {out_jsonl}")
    print(f"[WROTE] metrics:     {metrics_json}")

#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors. Licensed under the
# Apache License, Version 2.0; see the LICENSE file or
# https://www.apache.org/licenses/LICENSE-2.0 for details.

"""Evaluation routines for the GPT-4o slate-ranking baseline.

Fetches the configured dataset, issues batched GPT-4o requests, parses the
responses, and aggregates accuracy and formatting metrics for reporting.
"""

# pylint: disable=too-many-branches,too-many-locals,too-many-statements,broad-exception-caught,duplicate-code

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from common.eval_utils import ensure_hf_cache
from common.hf_datasets import get_dataset_loaders, require_dataset_support
from common.slate_eval import (
    EvaluationAccumulator,
    EvaluationFilters,
    Observation,
    bucket_from_options,
    bucket_from_position,
)

from .client import ds_call
from .config import DATASET_NAME, DEPLOYMENT_NAME, EVAL_SPLIT
from .conversation import make_conversation_record
from .utils import ANS_TAG, INDEX_ONLY

if TYPE_CHECKING:  # pragma: no cover - typing only imports
    from argparse import Namespace
else:  # pragma: no cover - runtime fallback for type hints
    Namespace = object  # type: ignore[misc]

DownloadConfig, load_dataset, load_from_disk = get_dataset_loaders()


def _parse_index_from_output(raw: str) -> int | None:
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


def run_eval(args: Namespace) -> None:
    """
    Evaluate GPT-4o on the configured dataset.

    :param args: Namespace with CLI parameters (temperature, max_tokens, eval_max, etc.)
    :type args: Namespace
    """

    dataset_name = str(getattr(args, "dataset", "") or DATASET_NAME)
    logging.info("Loading dataset %s", dataset_name)
    out_dir = Path(args.out_dir)
    _ensure_output_dir(out_dir, args.overwrite)
    out_jsonl = out_dir / "predictions.jsonl"
    metrics_json = out_dir / "metrics.json"

    ensure_hf_cache(args.cache_dir)

    dataset_path = Path(dataset_name)
    if dataset_path.exists():
        require_dataset_support(needs_local=True)
        assert load_from_disk is not None  # narrow Optional for type checkers
    else:
        require_dataset_support()
        assert load_dataset is not None and DownloadConfig is not None

    use_streaming = False
    dataset = None
    if dataset_path.exists():
        logging.info("Detected local dataset at %s", dataset_path)
        dataset = load_from_disk(str(dataset_path))  # type: ignore[arg-type]
    else:
        download_config = DownloadConfig(resume_download=True, max_retries=2)  # type: ignore[misc]
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

    assert load_dataset is not None  # for optional dependency imports
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

            pos_bucket = bucket_from_position(position_index)

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
            option_bucket = bucket_from_options(option_count)

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
        EvaluationFilters(issues=requested_issues, studies=requested_studies),
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

"""Evaluation loop for the GPT-4o slate baseline."""

# pylint: disable=too-many-branches,too-many-locals,too-many-statements,broad-exception-caught

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from datasets import DownloadConfig, load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DownloadConfig = None  # type: ignore
    load_dataset = None  # type: ignore

from common.eval_utils import safe_div

from .client import ds_call
from .config import DATASET_NAME, DEPLOYMENT_NAME, EVAL_SPLIT
from .conversation import make_conversation_record
from .utils import ANS_TAG, INDEX_ONLY


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

    logging.info("Loading dataset %s", DATASET_NAME)
    out_dir = Path(args.out_dir)
    _ensure_output_dir(out_dir, args.overwrite)
    out_jsonl = out_dir / "predictions.jsonl"
    metrics_json = out_dir / "metrics.json"

    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    if load_dataset is None or DownloadConfig is None:
        raise ImportError(
            "The 'datasets' package is required to run GPT-4o evaluations. "
            "Install it with `pip install datasets`."
        )

    download_config = DownloadConfig(resume_download=True, max_retries=2)
    use_streaming = False
    try:
        dataset = load_dataset(
            DATASET_NAME,
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
            data_iter = load_dataset(DATASET_NAME, split=eval_split, streaming=True)
        except Exception as exc:
            for fallback in ("validation", "eval", "test"):
                try:
                    data_iter = load_dataset(DATASET_NAME, split=fallback, streaming=True)
                    eval_split = fallback
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("Unable to load evaluation split in streaming mode.") from exc
    else:
        available_splits = list(dataset.keys())
        eval_split = next(
            (
                split
                for split in (EVAL_SPLIT, "validation", "eval", "test")
                if split in available_splits
            ),
            None,
        )
        if not eval_split:
            available_msg = ", ".join(available_splits)
            raise RuntimeError(
                "Evaluation split not found in dataset. "
                f"Available: {available_msg}"
            )
        data_iter = dataset[eval_split]
        if args.eval_max:
            data_iter = data_iter.select(range(min(args.eval_max, len(data_iter))))

    if use_streaming and args.eval_max:
        data_iter = data_iter.take(args.eval_max)

    n_eval_target = args.eval_max if args.eval_max else None
    with open(out_jsonl, "w", encoding="utf-8") as writer:
        buckets = ["1", "2", "3", "4", "5+", "unknown"]
        opts_buckets = ["1", "2", "3", "4", "5+"]
        seen_by_pos = defaultdict(int)
        eligible_by_pos = defaultdict(int)
        correct_by_pos = defaultdict(int)

        seen_by_opts = defaultdict(int)
        eligible_by_opts = defaultdict(int)
        correct_by_opts = defaultdict(int)
        parsed_by_opts = defaultdict(int)
        formatted_by_opts = defaultdict(int)

        seen_single = seen_multi = 0
        elig_single = elig_multi = 0
        corr_single = corr_multi = 0
        parsed_multi = formatted_multi = 0

        eligible_overall = correct_overall = 0
        format_ok = parsed_ok = 0

        gold_hist: Dict[int, int] = {}
        all_gold_indices: list[int] = []
        all_option_counts: list[int] = []

        start_time = time.time()
        seen_rows = 0

        for example in data_iter:
            seen_rows += 1
            record = make_conversation_record(example)
            messages = record["prompt"]
            gold_index = int(record.get("gold_index", -1))
            option_count = int(record.get("n_options", 0))
            position_index = int(record.get("position_index", -1))

            pos_bucket = _bucket_from_position(position_index)
            seen_by_pos[pos_bucket] += 1

            try:
                raw_output = ds_call(
                    messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    deployment=getattr(args, "deployment", None),
                )
            except Exception as exc:
                raw_output = f"(error: {exc})"

            is_formatted = bool(ANS_TAG.search(raw_output))
            if is_formatted:
                format_ok += 1

            parsed_index = _parse_index_from_output(raw_output)
            if parsed_index is not None:
                parsed_ok += 1

            option_bucket = _bucket_from_options(option_count)
            seen_by_opts[option_bucket] += 1
            if is_formatted:
                formatted_by_opts[option_bucket] += 1
            if parsed_index is not None:
                parsed_by_opts[option_bucket] += 1

            if option_count == 1:
                seen_single += 1
            else:
                seen_multi += 1
                if is_formatted:
                    formatted_multi += 1
                if parsed_index is not None:
                    parsed_multi += 1

            eligible = gold_index > 0 and option_count > 0
            if eligible:
                eligible_overall += 1
                eligible_by_pos[pos_bucket] += 1
                eligible_by_opts[option_bucket] += 1

                gold_hist[gold_index] = gold_hist.get(gold_index, 0) + 1
                all_gold_indices.append(gold_index)
                all_option_counts.append(option_count)

            is_correct = (
                eligible and (parsed_index is not None) and (parsed_index == gold_index)
            )
            if is_correct:
                correct_overall += 1
                correct_by_pos[pos_bucket] += 1
                correct_by_opts[option_bucket] += 1

            if eligible:
                if option_count == 1:
                    elig_single += 1
                    if is_correct:
                        corr_single += 1
                else:
                    elig_multi += 1
                    if is_correct:
                        corr_multi += 1

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
                        "position_index": position_index,
                        "position_bucket": pos_bucket,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            if seen_rows % 25 == 0:
                elapsed = time.time() - start_time
                accuracy = safe_div(correct_overall, eligible_overall)
                denom = n_eval_target if n_eval_target is not None else seen_rows
                logging.info(
                    "[eval] %d/%s  acc=%.3f  parsed=%.3f  fmt=%.3f  %.1fs",
                    seen_rows,
                    denom,
                    accuracy,
                    safe_div(parsed_ok, seen_rows),
                    safe_div(format_ok, seen_rows),
                    elapsed,
                )

    total_seen = seen_rows
    overall_accuracy = safe_div(correct_overall, eligible_overall)
    format_rate = safe_div(format_ok, total_seen)
    parsed_rate = safe_div(parsed_ok, total_seen)

    position_stats = {
        bucket: {
            "n_seen": int(seen_by_pos[bucket]),
            "n_eligible": int(eligible_by_pos[bucket]),
            "correct": int(correct_by_pos[bucket]),
            "accuracy": safe_div(correct_by_pos[bucket], eligible_by_pos[bucket]),
        }
        for bucket in buckets
    }

    by_options = {
        "hist_seen": {bucket: int(seen_by_opts[bucket]) for bucket in opts_buckets},
        "hist_eligible": {bucket: int(eligible_by_opts[bucket]) for bucket in opts_buckets},
        "hist_correct": {bucket: int(correct_by_opts[bucket]) for bucket in opts_buckets},
        "accuracy": {
            bucket: safe_div(correct_by_opts[bucket], eligible_by_opts[bucket])
            for bucket in opts_buckets
        },
        "parsed_rate": {
            bucket: safe_div(parsed_by_opts[bucket], seen_by_opts[bucket])
            for bucket in opts_buckets
        },
        "format_rate": {
            bucket: safe_div(formatted_by_opts[bucket], seen_by_opts[bucket])
            for bucket in opts_buckets
        },
    }

    split_single_multi = {
        "n_single": int(seen_single),
        "n_multi": int(seen_multi),
        "eligible_single": int(elig_single),
        "eligible_multi": int(elig_multi),
        "accuracy_single": safe_div(corr_single, elig_single),
        "accuracy_multi": safe_div(corr_multi, elig_multi),
        "parsed_rate_multi": safe_div(parsed_multi, max(1, seen_multi)),
        "format_rate_multi": safe_div(formatted_multi, max(1, seen_multi)),
    }

    gold_index_distribution = {str(key): int(value) for key, value in sorted(gold_hist.items())}
    if gold_hist:
        top_index = max(gold_hist.items(), key=lambda kv: kv[1])[0]
        baseline_correct = sum(1 for idx in all_gold_indices if idx == top_index)
        baseline_accuracy = safe_div(baseline_correct, eligible_overall)
    else:
        top_index = None
        baseline_accuracy = 0.0

    baseline_most_frequent = {
        "top_index": top_index,
        "count": int(gold_hist.get(top_index, 0) if top_index is not None else 0),
        "accuracy": baseline_accuracy,
    }

    random_baseline_accuracy = (
        float(np.mean([1.0 / n for n in all_option_counts]))
        if all_option_counts
        else 0.0
    )

    metrics: Dict[str, Any] = {
        "model": getattr(args, "deployment", None) or DEPLOYMENT_NAME,
        "dataset": DATASET_NAME,
        "split": eval_split,
        "n_total": int(total_seen),
        "n_eligible": int(eligible_overall),
        "accuracy_overall": overall_accuracy,
        "parsed_rate": parsed_rate,
        "format_rate": format_rate,
        "position_stats": position_stats,
        "by_n_options": by_options,
        "split_single_vs_multi": split_single_multi,
        "gold_index_distribution": gold_index_distribution,
        "baseline_most_frequent_gold_index": baseline_most_frequent,
        "random_baseline_expected_accuracy": random_baseline_accuracy,
        "notes": "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.",
    }

    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    summary = (
        f"[DONE] split={eval_split}  n={total_seen}  eligible={eligible_overall} "
        f"accuracy={overall_accuracy:.4f}  parsed_ok={parsed_rate:.3f}  "
        f"format_rate={format_rate:.3f}"
    )
    print(summary)
    print(f"[WROTE] per-example: {out_jsonl}")
    print(f"[WROTE] metrics:     {metrics_json}")

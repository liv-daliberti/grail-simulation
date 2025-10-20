"""Evaluation loop and metrics for the KNN baseline."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    SOLUTION_COLUMN,
    TRAIN_SPLIT,
    filter_dataset_for_issue,
    issues_in_dataset,
    load_dataset_source,
)
from .features import extract_slate_items
from .index import (
    build_tfidf_index,
    knn_predict_among_slate_multi,
    load_tfidf_index,
    save_tfidf_index,
)


def parse_k_values(k_default: int, sweep: str) -> List[int]:
    """Return the sorted list of k values derived from CLI configuration."""

    values = {int(k_default)} if k_default else set()
    for token in sweep.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            continue
    k_vals = sorted(k for k in values if k > 0)
    return k_vals or [int(k_default) if k_default else 25]


def select_best_k(k_values: Sequence[int], accuracy_by_k: Dict[int, float]) -> int:
    """Return the elbow-selected ``k`` from accuracy measurements."""

    if len(k_values) <= 2:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    accuracies = [accuracy_by_k.get(k, 0.0) for k in k_values]
    slopes: List[float] = []
    for idx in range(1, len(k_values)):
        delta_acc = accuracies[idx] - accuracies[idx - 1]
        delta_k = k_values[idx] - k_values[idx - 1]
        slopes.append(delta_acc / delta_k if delta_k else 0.0)
    if not slopes:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    first_slope = slopes[0]
    threshold = max(first_slope * 0.5, 0.001)
    for idx, slope in enumerate(slopes[1:], start=1):
        if slope <= threshold:
            return k_values[idx]
    return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))


def resolve_reports_dir(out_dir: Path) -> Path:
    """Return the reports directory associated with ``out_dir``."""

    resolved = out_dir.resolve()
    parents = list(resolved.parents)
    if len(parents) >= 1 and parents[0].name == "knn":
        resolved = parents[0]
        parents = list(resolved.parents)
    if len(parents) >= 1 and parents[0].name == "models":
        root_dir = parents[0].parent
    elif len(parents) >= 2 and parents[1].name == "models":
        root_dir = parents[1].parent
    else:
        root_dir = resolved.parent
    return root_dir / "reports"


def plot_elbow(k_values: Sequence[int], accuracy_by_k: Dict[int, float], best_k: int, output_path: Path) -> None:
    """Create an accuracy vs k plot for diagnostic purposes."""

    if plt is None:
        logging.warning("[KNN] Skipping elbow plot (matplotlib not installed)")
        return

    plt.figure(figsize=(6, 4))
    ys = [accuracy_by_k.get(k, 0.0) for k in k_values]
    plt.plot(k_values, ys, marker="o", label="Accuracy")
    if best_k in accuracy_by_k:
        plt.scatter([best_k], [accuracy_by_k[best_k]], color="red", label=f"Best k={best_k}")
    plt.title("KNN accuracy vs k")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def safe_div(numerator: float, denominator: float) -> float:
    """Return the division result guarding against a zero denominator."""

    return numerator / denominator if denominator else 0.0


def run_eval(args) -> None:
    """Evaluate the KNN baseline across the requested issues."""

    os_env = os.environ
    os_env.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os_env.setdefault("HF_HOME", args.cache_dir)

    dataset_source = args.dataset or DEFAULT_DATASET_SOURCE
    base_ds = load_dataset_source(dataset_source, args.cache_dir)
    available_issues = issues_in_dataset(base_ds)

    if args.issues:
        requested = [
            token.strip()
            for token in args.issues.split(",")
            if token.strip()
        ]
        issues = requested if requested else available_issues
    else:
        issues = available_issues

    k_values = parse_k_values(args.knn_k, args.knn_k_sweep)
    logging.info("[KNN] Evaluating k values: %s", k_values)

    for issue in issues:
        issue_slug = issue.replace(" ", "_")
        ds = filter_dataset_for_issue(base_ds, issue)
        train_ds = ds[TRAIN_SPLIT]
        eval_ds = ds[EVAL_SPLIT]

        extra_fields = [
            token.strip()
            for token in (args.knn_text_fields or "").split(",")
            if token.strip()
        ]
        if args.fit_index:
            logging.info("[KNN] Building TF-IDF index for issue=%s", issue_slug)
            knn_index = build_tfidf_index(
                train_ds,
                max_train=args.knn_max_train,
                seed=args.knn_seed,
                max_features=None,
                extra_fields=extra_fields,
            )
            if args.save_index:
                save_tfidf_index(knn_index, Path(args.save_index) / issue_slug)
        elif args.load_index:
            logging.info("[KNN] Loading TF-IDF index for issue=%s", issue_slug)
            knn_index = load_tfidf_index(Path(args.load_index) / issue_slug)
        else:
            raise ValueError("Set either --fit_index or --load_index to obtain a KNN index")

        evaluate_issue(
            issue_slug=issue_slug,
            dataset_source=dataset_source,
            eval_ds=eval_ds,
            k_values=k_values,
            knn_index=knn_index,
            extra_fields=extra_fields,
            args=args,
        )


def evaluate_issue(
    *,
    issue_slug: str,
    dataset_source: str,
    eval_ds,
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    extra_fields: Sequence[str],
    args,
) -> None:  # pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements
    """Evaluate a single issue split and write metrics/predictions."""
    indices = list(range(len(eval_ds)))
    if args.eval_max and args.eval_max > 0:
        indices = indices[: args.eval_max]

    rows: List[Dict[str, Any]] = []
    gold_hist: Dict[int, int] = {}
    buckets = ["unknown", "1", "2", "3", "4", "5+"]
    pos_stats = {b: 0 for b in buckets}
    corr_pos = {b: 0 for b in buckets}
    seen_opts = {b: 0 for b in buckets}
    elig_opts = {b: 0 for b in buckets}
    corr_opts = {b: 0 for b in buckets}

    eligible_by_k = {k: 0 for k in k_values}
    correct_by_k = {k: 0 for k in k_values}

    seen_single = seen_multi = 0
    elig_single = elig_multi = 0
    corr_single = corr_multi = 0

    all_gold_indices: List[int] = []
    all_n_options: List[int] = []

    start_time = time.time()

    for idx, row_index in enumerate(indices, start=1):
        example = eval_ds[int(row_index)]
        slate_pairs = extract_slate_items(example)
        n_options = len(slate_pairs)
        n_bucket = bin_nopts(n_options)
        seen_opts[n_bucket] += 1

        position_raw = example.get("video_index")
        try:
            position = int(position_raw) if position_raw is not None else -1
        except (TypeError, ValueError):
            position = -1
        pos_bucket = bucket_from_pos(position)
        pos_stats[pos_bucket] += 1

        gold_index = int(example.get("gold_index") or -1)
        gold_raw = str(example.get(SOLUTION_COLUMN, "")).strip()
        if gold_index < 1 and slate_pairs:
            for option_index, (title, vid) in enumerate(slate_pairs, start=1):
                if gold_raw and (gold_raw == vid or canon(gold_raw) == canon(title)):
                    gold_index = option_index
                    break

        predictions = knn_predict_among_slate_multi(
            knn_index=knn_index,
            example=example,
            k_values=k_values,
            text_fields=extra_fields,
            metric=args.knn_metric,
        )

        eligible = gold_index > 0 and n_options > 0
        if eligible:
            elig_opts[n_bucket] += 1
            gold_hist[gold_index] = gold_hist.get(gold_index, 0) + 1
            all_gold_indices.append(gold_index)
            all_n_options.append(n_options)
            if n_options == 1:
                elig_single += 1
            else:
                elig_multi += 1
        if n_options == 1:
            seen_single += 1
        elif n_options > 1:
            seen_multi += 1

        for k, pred in predictions.items():
            if eligible:
                eligible_by_k[k] += 1
                if pred is not None and int(pred) == gold_index:
                    correct_by_k[k] += 1

        rows.append(
            {
                "predictions_by_k": predictions,
                "gold_index": int(gold_index),
                "n_options": int(n_options),
                "n_options_bucket": n_bucket,
                "eligible": bool(eligible),
                "position_index": int(position),
                "position_bucket": pos_bucket,
                "issue_value": example.get("issue"),
            }
        )

        if idx % 25 == 0:
            elapsed = time.time() - start_time
            interim = max(
                (safe_div(correct_by_k[k], eligible_by_k[k]) for k in k_values if eligible_by_k[k]),
                default=0.0,
            )
            logging.info(
                "[eval][%s] %d/%d  interim-best-acc=%.3f  %.1fs",
                issue_slug,
                idx,
                len(indices),
                interim,
                elapsed,
            )

    accuracy_by_k = {k: safe_div(correct_by_k[k], eligible_by_k[k]) for k in k_values}
    best_k = select_best_k(k_values, accuracy_by_k)
    best_accuracy = accuracy_by_k.get(best_k, 0.0)
    eligible_overall = int(eligible_by_k.get(best_k, 0))

    for row in rows:
        if not row["eligible"]:
            continue
        prediction = row["predictions_by_k"].get(best_k)
        if prediction is None:
            continue
        if int(prediction) == row["gold_index"]:
            corr_pos[row["position_bucket"]] += 1
            corr_opts[row["n_options_bucket"]] += 1
            if row["n_options"] == 1:
                corr_single += 1
            else:
                corr_multi += 1

    out_dir = Path(args.out_dir) / issue_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"knn_eval_{issue_slug}_{EVAL_SPLIT}.jsonl"
    metrics_json = out_dir / f"knn_eval_{issue_slug}_{EVAL_SPLIT}_metrics.json"

    with open(out_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            preds_serializable = {
                str(k): (int(v) if v is not None else None)
                for k, v in row["predictions_by_k"].items()
            }
            best_pred = row["predictions_by_k"].get(best_k)
            record = {
                "knn_pred_index": int(best_pred) if best_pred is not None else None,
                "gold_index": row["gold_index"],
                "n_options": row["n_options"],
                "correct": bool(best_pred is not None and int(best_pred) == row["gold_index"]),
                "eligible": row["eligible"],
                "position_index": row["position_index"],
                "position_bucket": row["position_bucket"],
                "issue": issue_slug if issue_slug != "all" else row.get("issue_value"),
                "predictions_by_k": preds_serializable,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    pos_stats_out = {bucket: int(pos_stats[bucket]) for bucket in buckets}
    accuracy_opts_out = {bucket: safe_div(corr_opts[bucket], elig_opts[bucket]) for bucket in buckets}

    gold_distribution = {str(k): int(v) for k, v in sorted(gold_hist.items())}
    if gold_hist:
        top_idx = max(gold_hist.items(), key=lambda kv: kv[1])[0]
        baseline_correct = sum(1 for gi in all_gold_indices if gi == top_idx)
        baseline_accuracy = safe_div(baseline_correct, eligible_overall)
    else:
        top_idx = None
        baseline_accuracy = 0.0

    random_baseline = (
        float(np.mean([1.0 / n for n in all_n_options]))
        if all_n_options
        else 0.0
    )
    accuracy_by_k_serializable = {
        str(k): float(accuracy_by_k[k])
        for k in k_values
    }

    reports_dir = resolve_reports_dir(Path(args.out_dir)) / "knn"
    reports_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = reports_dir / f"elbow_{issue_slug}.png"
    plot_elbow(k_values, accuracy_by_k, best_k, elbow_path)

    metrics = {
        "model": "knn",
        "dataset": dataset_source,
        "issue": issue_slug,
        "split": EVAL_SPLIT,
        "n_total": int(len(indices)),
        "n_eligible": int(eligible_overall),
        "accuracy_overall": best_accuracy,
        "accuracy_by_k": accuracy_by_k_serializable,
        "best_k": int(best_k),
        "position_stats": pos_stats_out,
        "by_n_options": {
            bucket: {
                "hist_seen": int(seen_opts[bucket]),
                "hist_eligible": int(elig_opts[bucket]),
                "hist_correct": int(corr_opts[bucket]),
                "accuracy": accuracy_opts_out[bucket],
            }
            for bucket in buckets
        },
        "split_single_vs_multi": {
            "n_single": int(seen_single),
            "n_multi": int(seen_multi),
            "eligible_single": int(elig_single),
            "eligible_multi": int(elig_multi),
            "accuracy_single": safe_div(corr_single, elig_single),
            "accuracy_multi": safe_div(corr_multi, elig_multi),
        },
        "gold_index_distribution": gold_distribution,
        "baseline_most_frequent_gold_index": {
            "top_index": top_idx,
            "count": int(gold_hist.get(top_idx, 0) if top_idx is not None else 0),
            "accuracy": baseline_accuracy,
        },
        "random_baseline_expected_accuracy": random_baseline,
        "knn_hparams": {
            "k": int(args.knn_k),
            "k_sweep": [int(k) for k in k_values],
            "metric": args.knn_metric,
            "fit_index": bool(args.fit_index),
            "save_index": args.save_index or "",
            "load_index": args.load_index or "",
            "text_fields": list(extra_fields),
        },
        "elbow_plot": str(elbow_path),
        "notes": "Accuracy computed over eligible rows (gold_index>0).",
    }

    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    logging.info(
        "[DONE][%s] split=%s n=%d eligible=%d knn_acc=%.4f (best_k=%d)",
        issue_slug,
        EVAL_SPLIT,
        len(indices),
        eligible_overall,
        best_accuracy,
        best_k,
    )
    logging.info("[WROTE] per-example: %s", out_jsonl)
    logging.info("[WROTE] metrics: %s", metrics_json)


def bin_nopts(n: int) -> str:
    """Bucket the number of options into the reporting categories."""

    if n <= 1:
        return "1"
    if n == 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    return "5+"


def bucket_from_pos(pos_idx: int) -> str:
    """Bucket the original position index (0-based) into reporting bins."""

    if pos_idx < 0:
        return "unknown"
    if pos_idx == 0:
        return "1"
    if pos_idx == 1:
        return "2"
    if pos_idx == 2:
        return "3"
    if pos_idx == 3:
        return "4"
    return "5+"


def canon(text: str) -> str:
    """Canonicalise a text fragment for comparisons."""

    return re.sub(r"[^a-z0-9]+", "", (text or "").lower().strip())


__all__ = [
    "run_eval",
]

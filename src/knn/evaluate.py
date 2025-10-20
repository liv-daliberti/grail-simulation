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

from common.eval_utils import safe_div

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
from .features import Word2VecConfig, extract_slate_items
from .index import (
    SlateQueryConfig,
    build_tfidf_index,
    build_word2vec_index,
    knn_predict_among_slate_multi,
    load_tfidf_index,
    load_word2vec_index,
    save_tfidf_index,
    save_word2vec_index,
)


BUCKET_LABELS = ["unknown", "1", "2", "3", "4", "5+"]


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


def plot_elbow(
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    best_k: int,
    output_path: Path,
) -> None:
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


def _normalise_feature_space(feature_space: str | None) -> str:
    """Return the validated feature space identifier."""

    value = (feature_space or "tfidf").lower()
    if value not in {"tfidf", "word2vec"}:
        raise ValueError(f"Unsupported feature space '{feature_space}'")
    return value


def _word2vec_config_from_args(args, issue_slug: str) -> Word2VecConfig:
    """Return the Word2Vec configuration derived from CLI arguments."""

    default_cfg = Word2VecConfig()
    model_root = Path(args.word2vec_model_dir) if args.word2vec_model_dir else default_cfg.model_dir
    return Word2VecConfig(
        vector_size=int(args.word2vec_size),
        window=default_cfg.window,
        min_count=default_cfg.min_count,
        epochs=default_cfg.epochs,
        model_dir=Path(model_root) / issue_slug,
    )


def _fit_index_for_issue(
    *,
    feature_space: str,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
):
    """Build an index for the requested feature space and handle persistence."""

    if feature_space == "tfidf":
        logging.info("[KNN] Building TF-IDF index for issue=%s", issue_slug)
        index = build_tfidf_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            max_features=None,
            extra_fields=extra_fields,
        )
        if args.save_index:
            save_tfidf_index(index, Path(args.save_index) / issue_slug)
        return index

    if feature_space == "word2vec":
        logging.info("[KNN] Building Word2Vec index for issue=%s", issue_slug)
        config = _word2vec_config_from_args(args, issue_slug)
        index = build_word2vec_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            extra_fields=extra_fields,
            config=config,
        )
        if args.save_index:
            save_word2vec_index(index, Path(args.save_index) / issue_slug)
        return index

    raise ValueError(f"Unsupported feature space '{feature_space}'")


def _load_index_for_issue(
    *,
    feature_space: str,
    issue_slug: str,
    args,
):
    """Load a persisted index for the requested feature space."""

    load_path = Path(args.load_index) / issue_slug
    if feature_space == "tfidf":
        logging.info("[KNN] Loading TF-IDF index for issue=%s", issue_slug)
        return load_tfidf_index(load_path)
    if feature_space == "word2vec":
        logging.info("[KNN] Loading Word2Vec index for issue=%s", issue_slug)
        return load_word2vec_index(load_path)
    raise ValueError(f"Unsupported feature space '{feature_space}'")


def _build_or_load_index(
    *,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
):
    """Return the KNN index for ``issue_slug`` based on CLI arguments."""

    feature_space = _normalise_feature_space(getattr(args, "feature_space", None))
    if args.fit_index:
        return _fit_index_for_issue(
            feature_space=feature_space,
            train_ds=train_ds,
            issue_slug=issue_slug,
            extra_fields=extra_fields,
            args=args,
        )
    if args.load_index:
        return _load_index_for_issue(
            feature_space=feature_space,
            issue_slug=issue_slug,
            args=args,
        )
    raise ValueError("Set either --fit_index or --load_index to obtain a KNN index")


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
        knn_index = _build_or_load_index(
            train_ds=train_ds,
            issue_slug=issue_slug,
            extra_fields=extra_fields,
            args=args,
        )
        feature_space = str(knn_index.get("feature_space", "tfidf")).lower()

        evaluate_issue(
            issue_slug=issue_slug,
            dataset_source=dataset_source,
            eval_ds=eval_ds,
            k_values=k_values,
            knn_index=knn_index,
            extra_fields=extra_fields,
            feature_space=feature_space,
            args=args,
        )


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements

def evaluate_issue(
    *,
    issue_slug: str,
    dataset_source: str,
    eval_ds,
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    extra_fields: Sequence[str],
    feature_space: str,
    args,
) -> None:
    """Evaluate a single issue split and write metrics/predictions."""
    k_values = [int(k) for k in k_values]
    indices = list(range(len(eval_ds)))
    if args.eval_max and args.eval_max > 0:
        indices = indices[: args.eval_max]

    rows: List[Dict[str, Any]] = []
    gold_hist: Dict[int, int] = {}
    bucket_stats = {
        "position_seen": {b: 0 for b in BUCKET_LABELS},
        "options_seen": {b: 0 for b in BUCKET_LABELS},
        "options_eligible": {b: 0 for b in BUCKET_LABELS},
        "options_correct": {b: 0 for b in BUCKET_LABELS},
    }

    per_k_stats = {k: {"eligible": 0, "correct": 0} for k in k_values}

    single_multi_stats = {
        "seen_single": 0,
        "seen_multi": 0,
        "elig_single": 0,
        "elig_multi": 0,
        "corr_single": 0,
        "corr_multi": 0,
    }

    all_gold_indices: List[int] = []
    all_n_options: List[int] = []

    start_time = time.time()

    query_config = SlateQueryConfig(
        text_fields=tuple(extra_fields),
        lowercase=True,
        metric=args.knn_metric,
    )

    for idx, row_index in enumerate(indices, start=1):
        example = eval_ds[int(row_index)]
        slate_pairs = extract_slate_items(example)
        n_options = len(slate_pairs)
        n_bucket = bin_nopts(n_options)
        bucket_stats["options_seen"][n_bucket] += 1

        position_raw = example.get("video_index")
        try:
            position = int(position_raw) if position_raw is not None else -1
        except (TypeError, ValueError):
            position = -1
        pos_bucket = bucket_from_pos(position)
        bucket_stats["position_seen"][pos_bucket] += 1

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
            config=query_config,
        )

        eligible = gold_index > 0 and n_options > 0
        if eligible:
            bucket_stats["options_eligible"][n_bucket] += 1
            gold_hist[gold_index] = gold_hist.get(gold_index, 0) + 1
            all_gold_indices.append(gold_index)
            all_n_options.append(n_options)
            if n_options == 1:
                single_multi_stats["elig_single"] += 1
            else:
                single_multi_stats["elig_multi"] += 1
        if n_options == 1:
            single_multi_stats["seen_single"] += 1
        elif n_options > 1:
            single_multi_stats["seen_multi"] += 1

        for k, pred in predictions.items():
            if eligible:
                k_stats = per_k_stats[k]
                k_stats["eligible"] += 1
                if pred is not None and int(pred) == gold_index:
                    k_stats["correct"] += 1

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
                (
                    safe_div(per_k_stats[k]["correct"], per_k_stats[k]["eligible"])
                    for k in k_values
                    if per_k_stats[k]["eligible"]
                ),
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

    accuracy_by_k = {
        k: safe_div(per_k_stats[k]["correct"], per_k_stats[k]["eligible"])
        for k in k_values
    }
    best_k = select_best_k(k_values, accuracy_by_k)
    best_accuracy = accuracy_by_k.get(best_k, 0.0)
    eligible_overall = int(per_k_stats.get(best_k, {}).get("eligible", 0))

    for row in rows:
        if not row["eligible"]:
            continue
        prediction = row["predictions_by_k"].get(best_k)
        if prediction is None:
            continue
        if int(prediction) == row["gold_index"]:
            bucket_stats["options_correct"][row["n_options_bucket"]] += 1
            if row["n_options"] == 1:
                single_multi_stats["corr_single"] += 1
            else:
                single_multi_stats["corr_multi"] += 1

    issue_dir = Path(args.out_dir) / issue_slug
    issue_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = issue_dir / f"knn_eval_{issue_slug}_{EVAL_SPLIT}.jsonl"
    metrics_json = issue_dir / f"knn_eval_{issue_slug}_{EVAL_SPLIT}_metrics.json"

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

    pos_stats_out = {
        bucket: int(bucket_stats["position_seen"][bucket])
        for bucket in BUCKET_LABELS
    }
    accuracy_opts_out = {
        bucket: safe_div(
            bucket_stats["options_correct"][bucket],
            bucket_stats["options_eligible"][bucket],
        )
        for bucket in BUCKET_LABELS
    }

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

    reports_dir = resolve_reports_dir(Path(args.out_dir)) / "knn" / feature_space
    reports_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = reports_dir / f"elbow_{issue_slug}.png"
    plot_elbow(k_values, accuracy_by_k, best_k, elbow_path)

    metrics = {
        "model": "knn",
        "feature_space": feature_space,
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
                "hist_seen": int(bucket_stats["options_seen"][bucket]),
                "hist_eligible": int(bucket_stats["options_eligible"][bucket]),
                "hist_correct": int(bucket_stats["options_correct"][bucket]),
                "accuracy": accuracy_opts_out[bucket],
            }
            for bucket in BUCKET_LABELS
        },
        "split_single_vs_multi": {
            "n_single": int(single_multi_stats["seen_single"]),
            "n_multi": int(single_multi_stats["seen_multi"]),
            "eligible_single": int(single_multi_stats["elig_single"]),
            "eligible_multi": int(single_multi_stats["elig_multi"]),
            "accuracy_single": safe_div(
                single_multi_stats["corr_single"],
                single_multi_stats["elig_single"],
            ),
            "accuracy_multi": safe_div(
                single_multi_stats["corr_multi"],
                single_multi_stats["elig_multi"],
            ),
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
        "per_k_directories": {
            str(k): str((issue_dir / f"k-{int(k)}"))
            for k in k_values
        },
        "notes": "Accuracy computed over eligible rows (gold_index>0).",
    }

    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    for k in k_values:
        k_int = int(k)
        k_dir = issue_dir / f"k-{k_int}"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_predictions_path = k_dir / f"predictions_{issue_slug}_{EVAL_SPLIT}.jsonl"
        with open(k_predictions_path, "w", encoding="utf-8") as handle:
            for row in rows:
                pred_value = row["predictions_by_k"].get(k)
                record = {
                    "k": k_int,
                    "knn_pred_index": int(pred_value) if pred_value is not None else None,
                    "gold_index": row["gold_index"],
                    "eligible": row["eligible"],
                    "correct": bool(pred_value is not None and int(pred_value) == row["gold_index"]),
                    "n_options": row["n_options"],
                    "position_index": row["position_index"],
                    "issue": issue_slug if issue_slug != "all" else row.get("issue_value"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        k_stats = per_k_stats[k_int]
        k_metrics = {
            "model": "knn",
            "feature_space": feature_space,
            "dataset": dataset_source,
            "issue": issue_slug,
            "split": EVAL_SPLIT,
            "k": k_int,
            "n_total": int(len(indices)),
            "n_eligible": int(k_stats["eligible"]),
            "n_correct": int(k_stats["correct"]),
            "accuracy": float(accuracy_by_k.get(k_int, 0.0)),
            "elbow_plot": str(elbow_path),
        }
        with open(k_dir / f"metrics_{issue_slug}_{EVAL_SPLIT}.json", "w", encoding="utf-8") as handle:
            json.dump(k_metrics, handle, ensure_ascii=False, indent=2)

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

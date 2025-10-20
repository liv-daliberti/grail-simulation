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
    *,
    data_split: str = "validation",
) -> None:
    """Create an error-rate vs ``k`` plot for diagnostic purposes.

    :param k_values: Iterable of evaluated ``k`` values.
    :param accuracy_by_k: Mapping from ``k`` to accuracy on the requested split.
    :param best_k: Selected ``k`` value for reporting.
    :param output_path: Destination path for the generated figure.
    :param data_split: Human-readable label describing the data split used.
    """

    if plt is None:
        logging.warning("[KNN] Skipping elbow plot (matplotlib not installed)")
        return

    if not k_values:
        logging.warning("[KNN] Skipping elbow plot (no k values supplied)")
        return

    plt.figure(figsize=(6, 4))
    error_rates = [1.0 - float(accuracy_by_k.get(k, 0.0)) for k in k_values]
    plt.plot(k_values, error_rates, marker="o", label="Error rate")
    if best_k in accuracy_by_k:
        best_error = 1.0 - float(accuracy_by_k[best_k])
        plt.axvline(best_k, color="red", linestyle="--", alpha=0.6)
        plt.scatter([best_k], [best_error], color="red", label="Selected k")
    split_label = data_split.strip() or "validation"
    plt.title(f"KNN error vs k ({split_label} split)")
    plt.xlabel("k")
    plt.ylabel("Error rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels)
    plt.figtext(
        0.5,
        -0.05,
        f"Error computed on {split_label} data (eligible examples only)",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def compute_auc_from_curve(k_values: Sequence[int], accuracy_by_k: Dict[int, float]) -> tuple[float, float]:
    """Return raw and normalised area under the accuracy vs k curve.

    :param k_values: Iterable of ``k`` values considered during evaluation.
    :param accuracy_by_k: Mapping from ``k`` to observed accuracy.
    :returns: Tuple of ``(auc_area, auc_normalized)`` where ``auc_area`` is the
        trapezoidal integral and ``auc_normalized`` divides by the span of ``k``.
    """

    if not k_values:
        return 0.0, 0.0
    sorted_k = sorted({int(k) for k in k_values})
    ys = [float(accuracy_by_k.get(k, 0.0)) for k in sorted_k]
    if len(sorted_k) == 1:
        value = ys[0]
        return value, value
    area = float(np.trapz(ys, sorted_k))
    span = float(sorted_k[-1] - sorted_k[0]) or 1.0
    return area, area / span


def _normalise_feature_space(feature_space: str | None) -> str:
    """Return the validated feature space identifier.

    :param feature_space: Raw feature-space string supplied via CLI.
    :returns: Lowercase feature-space token (``tfidf`` or ``word2vec``).
    :raises ValueError: If an unsupported feature space is supplied.
    """

    value = (feature_space or "tfidf").lower()
    if value not in {"tfidf", "word2vec"}:
        raise ValueError(f"Unsupported feature space '{feature_space}'")
    return value


def _word2vec_config_from_args(args, issue_slug: str) -> Word2VecConfig:
    """Return the Word2Vec configuration derived from CLI arguments.

    :param args: Parsed CLI namespace containing Word2Vec options.
    :param issue_slug: Current issue being processed (used to namespace models).
    :returns: Populated :class:`~knn.features.Word2VecConfig` instance.
    """

    default_cfg = Word2VecConfig()
    model_root = Path(args.word2vec_model_dir) if args.word2vec_model_dir else default_cfg.model_dir
    return Word2VecConfig(
        vector_size=int(args.word2vec_size),
        window=int(getattr(args, "word2vec_window", default_cfg.window)),
        min_count=int(getattr(args, "word2vec_min_count", default_cfg.min_count)),
        epochs=int(getattr(args, "word2vec_epochs", default_cfg.epochs)),
        model_dir=Path(model_root) / issue_slug,
        seed=int(getattr(args, "knn_seed", default_cfg.seed)),
        workers=int(getattr(args, "word2vec_workers", default_cfg.workers)),
    )


def _fit_index_for_issue(
    *,
    feature_space: str,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
):
    """Build an index for the requested feature space and handle persistence.

    :param feature_space: ``tfidf`` or ``word2vec``.
    :param train_ds: Training split (Hugging Face dataset slice).
    :param issue_slug: Normalised issue identifier.
    :param extra_fields: Optional text fields to concatenate into documents.
    :param args: CLI namespace for additional parameters.
    :returns: Dictionary describing the fitted index artifacts.
    :raises ValueError: If the requested feature space is unsupported.
    """

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
    """Load a persisted index for the requested feature space.

    :param feature_space: ``tfidf`` or ``word2vec``.
    :param issue_slug: Normalised issue identifier.
    :param args: CLI namespace providing the ``--load-index`` directory.
    :returns: Dictionary with the loaded index artifacts.
    :raises ValueError: If the feature space is not recognised.
    """

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
    """Return the KNN index for ``issue_slug`` based on CLI arguments.

    :param train_ds: Training split dataset.
    :param issue_slug: Normalised issue identifier.
    :param extra_fields: Optional extra text fields.
    :param args: CLI namespace containing ``--fit-index`` or ``--load-index``.
    :returns: Dictionary describing the fitted or loaded KNN index.
    :raises ValueError: When neither ``--fit-index`` nor ``--load-index`` is used.
    """

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


def run_eval(args) -> None:  # pylint: disable=too-many-locals
    """Evaluate the KNN baseline across the requested issues.

    :param args: Parsed CLI namespace returned by :func:`knn.cli.build_parser`.
    """

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
            train_ds=train_ds,
            eval_ds=eval_ds,
            k_values=k_values,
            knn_index=knn_index,
            extra_fields=extra_fields,
            feature_space=feature_space,
            args=args,
        )


# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements


def _write_issue_outputs(
    *,
    args,
    issue_slug: str,
    feature_space: str,
    dataset_source: str,
    rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    best_k: int,
    bucket_stats: Dict[str, Dict[str, int]],
    single_multi_stats: Dict[str, int],
    gold_hist: Dict[int, int],
    per_k_stats: Dict[int, Dict[str, int]],
    extra_fields: Sequence[str],
    curve_metrics: Dict[str, Any],
) -> None:
    """Persist evaluation artifacts and per-``k`` directories for an issue.

    :param args: CLI namespace controlling output directories.
    :param issue_slug: Issue slug associated with the evaluation batch.
    :param feature_space: Active feature space (``tfidf`` or ``word2vec``).
    :param dataset_source: Source dataset label written to metrics.
    :param rows: Per-example records produced during evaluation.
    :param k_values: Sequence of ``k`` values scored for the issue.
    :param accuracy_by_k: Accuracy measured for each ``k``.
    :param best_k: Elbow-selected ``k`` value.
    :param bucket_stats: Aggregated slate-position statistics.
    :param single_multi_stats: Aggregated single vs multi option metrics.
    :param gold_hist: Histogram of gold indices encountered.
    :param per_k_stats: Eligibility and correctness tallies per ``k``.
    :param extra_fields: Extra text fields contributing to the query document.
    :param curve_metrics: Serialised evaluation/train curve diagnostics.
    """

    best_accuracy = accuracy_by_k.get(best_k, 0.0)
    eligible_overall = int(per_k_stats.get(best_k, {}).get("eligible", 0))

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
    if gold_hist and eligible_overall:
        top_idx, top_count = max(gold_hist.items(), key=lambda kv: kv[1])
        baseline_accuracy = safe_div(top_count, eligible_overall)
    else:
        top_idx = None
        baseline_accuracy = 0.0

    random_sum = float(single_multi_stats.get("rand_inverse_sum", 0.0))
    random_count = int(single_multi_stats.get("rand_inverse_count", 0))
    random_baseline = safe_div(random_sum, random_count)
    accuracy_by_k_serializable = {
        str(k): float(accuracy_by_k[k])
        for k in k_values
    }

    reports_dir = resolve_reports_dir(Path(args.out_dir)) / "knn" / feature_space
    reports_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = reports_dir / f"elbow_{issue_slug}.png"
    plot_elbow(k_values, accuracy_by_k, best_k, elbow_path, data_split=EVAL_SPLIT)

    curve_json = issue_dir / f"knn_curves_{issue_slug}.json"
    with open(curve_json, "w", encoding="utf-8") as handle:
        json.dump(curve_metrics, handle, ensure_ascii=False, indent=2)

    metrics = {
        "model": "knn",
        "feature_space": feature_space,
        "dataset": dataset_source,
        "issue": issue_slug,
        "split": EVAL_SPLIT,
        "n_total": int(len(rows)),
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
        "curve_metrics": curve_metrics,
        "curve_metrics_path": str(curve_json),
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
                pred_value = row["predictions_by_k"].get(k_int)
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
            "n_total": int(len(rows)),
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
        len(rows),
        eligible_overall,
        best_accuracy,
        best_k,
    )
    logging.info("[WROTE] per-example: %s", out_jsonl)
    logging.info("[WROTE] metrics: %s", metrics_json)
    logging.info("[WROTE] curves: %s", curve_json)
def _accumulate_row(
    *,
    example: Dict[str, Any],
    bucket_stats: Dict[str, Dict[str, int]],
    per_k_stats: Dict[int, Dict[str, int]],
    single_multi_stats: Dict[str, int],
    gold_hist: Dict[int, int],
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    query_config: SlateQueryConfig,
) -> Dict[str, Any]:
    """Process a single evaluation example and update aggregate statistics.

    :param example: Dataset row representing one recommendation slate.
    :param bucket_stats: Mutable dictionary tracking per-bucket counts.
    :param per_k_stats: Mutable dictionary storing eligible/correct tallies per ``k``.
    :param single_multi_stats: Mutable dictionary tracking single vs multi-option stats.
    :param gold_hist: Mutable histogram of observed gold indices.
    :param k_values: Sequence of ``k`` values to score.
    :param knn_index: Prepared KNN index artifacts.
    :param query_config: Configuration controlling query generation.
    :returns: Serialised per-example record including predictions for each ``k``.
    """

    slate_pairs = extract_slate_items(example)
    n_options = len(slate_pairs)
    n_bucket = bin_nopts(n_options)
    bucket_stats["options_seen"][n_bucket] += 1

    try:
        position = int(example.get("video_index") or -1)
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
        single_multi_stats["rand_inverse_sum"] += (1.0 / n_options) if n_options else 0.0
        single_multi_stats["rand_inverse_count"] += 1
        if n_options == 1:
            single_multi_stats["elig_single"] += 1
        else:
            single_multi_stats["elig_multi"] += 1
    if n_options == 1:
        single_multi_stats["seen_single"] += 1
    elif n_options > 1:
        single_multi_stats["seen_multi"] += 1

    if eligible:
        for k, pred in predictions.items():
            k_stats = per_k_stats[k]
            k_stats["eligible"] += 1
            if pred is not None and int(pred) == gold_index:
                k_stats["correct"] += 1

    return {
        "predictions_by_k": predictions,
        "gold_index": int(gold_index),
        "n_options": int(n_options),
        "n_options_bucket": n_bucket,
        "eligible": bool(eligible),
        "position_index": int(position),
        "position_bucket": pos_bucket,
        "issue_value": example.get("issue"),
    }


def _evaluate_dataset_split(
    *,
    dataset,
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    extra_fields: Sequence[str],
    metric: str,
    capture_rows: bool,
    log_label: str,
    max_examples: int | None,
    log_k: int | None = None,
) -> Dict[str, Any]:
    """Return aggregate statistics for ``dataset`` using the provided index.

    :param dataset: Dataset slice to iterate.
    :param k_values: Sequence of ``k`` values to evaluate.
    :param knn_index: Prepared KNN index artifacts.
    :param extra_fields: Extra text fields appended to the query.
    :param metric: Distance metric for candidate scoring (``l2`` or ``cosine``).
    :param capture_rows: Whether to retain per-example prediction rows.
    :param log_label: Label emitted in progress logs.
    :param max_examples: Optional limit on the number of processed examples.
    :param log_k: Optional ``k`` to report accuracy for in progress logs.
    :returns: Dictionary containing rows, aggregate stats, and counts.
    """

    rows: List[Dict[str, Any]] = [] if capture_rows else []
    gold_hist: Dict[int, int] = {}
    bucket_stats = {
        "position_seen": {b: 0 for b in BUCKET_LABELS},
        "options_seen": {b: 0 for b in BUCKET_LABELS},
        "options_eligible": {b: 0 for b in BUCKET_LABELS},
        "options_correct": {b: 0 for b in BUCKET_LABELS},
    }
    per_k_stats = {k: {"eligible": 0, "correct": 0} for k in k_values}
    single_multi_stats: Dict[str, float | int] = {
        "seen_single": 0,
        "seen_multi": 0,
        "elig_single": 0,
        "elig_multi": 0,
        "corr_single": 0,
        "corr_multi": 0,
        "rand_inverse_sum": 0.0,
        "rand_inverse_count": 0,
    }

    dataset_len = len(dataset)
    limit = dataset_len
    if max_examples is not None and max_examples > 0:
        limit = min(dataset_len, max_examples)

    query_config = SlateQueryConfig(
        text_fields=tuple(extra_fields),
        lowercase=True,
        metric=metric,
    )

    log_k_value: int | None = None
    if log_k:
        desired = int(log_k)
        if desired in per_k_stats:
            log_k_value = desired
        elif per_k_stats:
            log_k_value = min(per_k_stats.keys(), key=lambda key_k: abs(key_k - desired))

    start_time = time.time()

    for idx in range(int(limit)):
        row = _accumulate_row(
            example=dataset[int(idx)],
            bucket_stats=bucket_stats,
            per_k_stats=per_k_stats,
            single_multi_stats=single_multi_stats,  # type: ignore[arg-type]
            gold_hist=gold_hist,
            k_values=k_values,
            knn_index=knn_index,
            query_config=query_config,
        )
        if capture_rows:
            rows.append(row)
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - start_time
            acc_message = ""
            if log_k_value is not None:
                stats = per_k_stats[log_k_value]
                acc_message = f"  acc@{log_k_value}={safe_div(stats['correct'], stats['eligible']):.3f}"
            logging.info(
                "[%s] %d/%d  elapsed=%.1fs%s",
                log_label,
                idx + 1,
                limit,
                elapsed,
                acc_message,
            )

    return {
        "rows": rows,
        "bucket_stats": bucket_stats,
        "per_k_stats": per_k_stats,
        "single_multi_stats": single_multi_stats,
        "gold_hist": gold_hist,
        "n_examples": int(limit),
    }


def _update_correct_counts(
    rows: Sequence[Dict[str, Any]],
    best_k: int,
    bucket_stats: Dict[str, Dict[str, int]],
    single_multi_stats: Dict[str, int],
) -> None:
    """Update bucket-level correctness tallies for the selected ``best_k``.

    :param rows: Iterable of per-example prediction records.
    :param best_k: Elbow-selected ``k`` used to judge correctness.
    :param bucket_stats: Mutable dictionary storing per-bucket correctness.
    :param single_multi_stats: Mutable dictionary tracking single vs multi counts.
    """

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


def _curve_summary(
    *,
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    per_k_stats: Dict[int, Dict[str, int]],
    best_k: int,
    n_examples: int,
) -> Dict[str, Any]:
    """Return a serialisable summary for accuracy-vs-k curves."""

    area, normalised = compute_auc_from_curve(k_values, accuracy_by_k)
    sorted_k = sorted({int(k) for k in k_values})
    accuracy_serialised = {
        str(k): float(accuracy_by_k.get(k, 0.0))
        for k in sorted_k
    }
    eligible_serialised = {
        str(k): int(per_k_stats[k]["eligible"])
        for k in sorted_k
    }
    correct_serialised = {
        str(k): int(per_k_stats[k]["correct"])
        for k in sorted_k
    }
    return {
        "accuracy_by_k": accuracy_serialised,
        "eligible_by_k": eligible_serialised,
        "correct_by_k": correct_serialised,
        "auc_area": float(area),
        "auc_normalized": float(normalised),
        "best_k": int(best_k),
        "best_accuracy": float(accuracy_by_k.get(best_k, 0.0)),
        "n_examples": int(n_examples),
    }


def evaluate_issue(
    *,
    issue_slug: str,
    dataset_source: str,
    train_ds,
    eval_ds,
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    extra_fields: Sequence[str],
    feature_space: str,
    args,
) -> None:  # pylint: disable=too-many-locals
    """Evaluate a single issue split and write metrics/predictions.

    :param issue_slug: Normalised issue identifier.
    :param dataset_source: Name or path of the dataset originated from.
    :param train_ds: Training split dataset for optional curve diagnostics.
    :param eval_ds: Evaluation split dataset.
    :param k_values: Sequence of ``k`` values to assess.
    :param knn_index: Prepared KNN index artifacts.
    :param extra_fields: Extra text fields appended to queries.
    :param feature_space: Active feature space (``tfidf`` or ``word2vec``).
    :param args: CLI namespace controlling evaluation options.
    """

    k_values_int = sorted({int(k) for k in k_values if int(k) > 0})
    eval_max = args.eval_max if args.eval_max and args.eval_max > 0 else None
    eval_summary = _evaluate_dataset_split(
        dataset=eval_ds,
        k_values=k_values_int,
        knn_index=knn_index,
        extra_fields=extra_fields,
        metric=args.knn_metric,
        capture_rows=True,
        log_label=f"eval][{issue_slug}",
        max_examples=eval_max,
        log_k=args.knn_k,
    )

    rows: List[Dict[str, Any]] = eval_summary["rows"]
    bucket_stats: Dict[str, Dict[str, int]] = eval_summary["bucket_stats"]
    single_multi_stats: Dict[str, int] = eval_summary["single_multi_stats"]  # type: ignore[assignment]
    gold_hist: Dict[int, int] = eval_summary["gold_hist"]
    per_k_stats: Dict[int, Dict[str, int]] = eval_summary["per_k_stats"]

    accuracy_by_k = {
        k: safe_div(per_k_stats[k]["correct"], per_k_stats[k]["eligible"])
        for k in k_values_int
    }
    best_k = select_best_k(k_values_int, accuracy_by_k)
    _update_correct_counts(rows, best_k, bucket_stats, single_multi_stats)
    eval_curve = _curve_summary(
        k_values=k_values_int,
        accuracy_by_k=accuracy_by_k,
        per_k_stats=per_k_stats,
        best_k=best_k,
        n_examples=eval_summary["n_examples"],
    )

    train_curve = None
    train_max = getattr(args, "train_curve_max", 0)
    if train_ds is not None:
        max_examples = train_max if train_max and train_max > 0 else None
        train_summary = _evaluate_dataset_split(
            dataset=train_ds,
            k_values=k_values_int,
            knn_index=knn_index,
            extra_fields=extra_fields,
            metric=args.knn_metric,
            capture_rows=False,
            log_label=f"train][{issue_slug}",
            max_examples=max_examples,
            log_k=args.knn_k,
        )
        train_accuracy_by_k = {
            k: safe_div(train_summary["per_k_stats"][k]["correct"], train_summary["per_k_stats"][k]["eligible"])
            for k in k_values_int
        }
        train_best_k = select_best_k(k_values_int, train_accuracy_by_k)
        train_curve = _curve_summary(
            k_values=k_values_int,
            accuracy_by_k=train_accuracy_by_k,
            per_k_stats=train_summary["per_k_stats"],
            best_k=train_best_k,
            n_examples=train_summary["n_examples"],
        )

    curve_metrics = {"eval": eval_curve}
    if train_curve:
        curve_metrics["train"] = train_curve

    _write_issue_outputs(
        args=args,
        issue_slug=issue_slug,
        feature_space=feature_space,
        dataset_source=dataset_source,
        rows=rows,
        k_values=k_values_int,
        accuracy_by_k=accuracy_by_k,
        best_k=best_k,
        bucket_stats=bucket_stats,
        single_multi_stats=single_multi_stats,
        gold_hist=gold_hist,
        per_k_stats=per_k_stats,
        extra_fields=extra_fields,
        curve_metrics=curve_metrics,
    )

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
    "parse_k_values",
    "select_best_k",
    "resolve_reports_dir",
    "plot_elbow",
    "compute_auc_from_curve",
    "evaluate_issue",
    "bin_nopts",
    "bucket_from_pos",
    "canon",
    "run_eval",
]

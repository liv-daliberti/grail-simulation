"""Post-study opinion index evaluation for the KNN baselines."""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.random import default_rng
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    load_dataset_source,
)
from .evaluate import parse_k_values, resolve_reports_dir
from .features import Word2VecConfig, Word2VecFeatureBuilder, assemble_document

try:  # pragma: no cover - optional dependency
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None


LOGGER = logging.getLogger("knn.opinion")


@dataclass(frozen=True)
class OpinionSpec:
    """Configuration describing one study's opinion index columns."""

    key: str
    issue: str
    label: str
    before_column: str
    after_column: str


@dataclass
class OpinionExample:
    """Collapsed participant-level prompt and opinion values."""

    participant_id: str
    participant_study: str
    issue: str
    document: str
    before: float
    after: float
    step_index: int
    session_id: Optional[str]


@dataclass
class OpinionIndex:
    """Vectorised training corpus and associated metadata."""

    feature_space: str
    metric: str
    matrix: Any
    vectorizer: Any
    embeds: Optional[Word2VecFeatureBuilder]
    targets_after: np.ndarray
    targets_before: np.ndarray
    participant_keys: List[Tuple[str, str]]
    neighbors: NearestNeighbors


DEFAULT_SPECS: Tuple[OpinionSpec, ...] = (
    OpinionSpec(
        key="study1",
        issue="gun_control",
        label="Study 1 – Gun Control (MTurk)",
        before_column="gun_index",
        after_column="gun_index_2",
    ),
    OpinionSpec(
        key="study2",
        issue="minimum_wage",
        label="Study 2 – Minimum Wage (MTurk)",
        before_column="mw_index_w1",
        after_column="mw_index_w2",
    ),
    OpinionSpec(
        key="study3",
        issue="minimum_wage",
        label="Study 3 – Minimum Wage (YouGov)",
        before_column="mw_index_w1",
        after_column="mw_index_w2",
    ),
)


def find_spec(key: str) -> OpinionSpec:
    """Return the :class:`OpinionSpec` matching ``key``."""

    normalised = key.strip().lower()
    for spec in DEFAULT_SPECS:
        if spec.key.lower() == normalised:
            return spec
    raise KeyError(f"Unknown opinion study '{key}'. Expected one of {[s.key for s in DEFAULT_SPECS]!r}")


def float_or_none(value: Any) -> Optional[float]:
    """Return ``value`` converted to ``float`` or ``None`` when invalid."""

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def collect_examples(
    dataset,
    *,
    spec: OpinionSpec,
    extra_fields: Sequence[str],
    max_examples: int | None,
    seed: int,
) -> List[OpinionExample]:
    """Collapse a split down to participant-level opinion rows."""

    LOGGER.info(
        "[OPINION] Collapsing dataset split for study=%s issue=%s rows=%d",
        spec.key,
        spec.issue,
        len(dataset),
    )
    per_participant: Dict[Tuple[str, str], OpinionExample] = {}

    for idx in range(len(dataset)):
        example = dataset[int(idx)]
        if example.get("issue") != spec.issue or example.get("participant_study") != spec.key:
            continue
        before = float_or_none(example.get(spec.before_column))
        after = float_or_none(example.get(spec.after_column))
        if before is None or after is None:
            continue
        document = assemble_document(example, extra_fields)
        if not document:
            continue
        try:
            step_index = int(example.get("step_index") or -1)
        except (TypeError, ValueError):
            step_index = -1
        participant_id = str(example.get("participant_id") or "")
        key = (participant_id, spec.key)
        existing = per_participant.get(key)
        session_id = example.get("session_id")
        candidate = OpinionExample(
            participant_id=participant_id,
            participant_study=spec.key,
            issue=spec.issue,
            document=document,
            before=before,
            after=after,
            step_index=step_index,
            session_id=str(session_id) if session_id is not None else None,
        )
        if existing is None or step_index >= existing.step_index:
            per_participant[key] = candidate

    collapsed = list(per_participant.values())
    LOGGER.info(
        "[OPINION] Collapsed to %d unique participants (from %d rows).",
        len(collapsed),
        len(dataset),
    )

    if max_examples and 0 < max_examples < len(collapsed):
        rng = default_rng(seed)
        order = rng.permutation(len(collapsed))[:max_examples]
        collapsed = [collapsed[i] for i in order]
        LOGGER.info("[OPINION] Sampled %d participants (max=%d).", len(collapsed), max_examples)

    return collapsed


def _build_tfidf_matrix(documents: Sequence[str]) -> Tuple[TfidfVectorizer, Any]:
    """Fit a TF-IDF vectoriser using the supplied documents."""

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,
        token_pattern=r"(?u)\b[\w\-]{2,}\b",
        max_features=None,
    )
    matrix = vectorizer.fit_transform(documents).astype(np.float32)
    return vectorizer, matrix


def build_index(
    *,
    examples: Sequence[OpinionExample],
    feature_space: str,
    spec: OpinionSpec,
    seed: int,
    metric: str,
    word2vec_config: Optional[Word2VecConfig] = None,
) -> OpinionIndex:
    """Vectorise ``examples`` and construct a neighbour index."""

    documents = [example.document for example in examples]
    feature_space = (feature_space or "tfidf").lower()

    if feature_space == "tfidf":
        vectorizer, matrix = _build_tfidf_matrix(documents)
        embeds = None
    elif feature_space == "word2vec":
        embeds = Word2VecFeatureBuilder(word2vec_config)
        embeds.train(documents)
        matrix = embeds.transform(documents)
        vectorizer = None
    else:
        raise ValueError(f"Unsupported feature space '{feature_space}'.")

    targets_after = np.asarray([example.after for example in examples], dtype=np.float32)
    targets_before = np.asarray([example.before for example in examples], dtype=np.float32)
    participant_keys = [(example.participant_id, example.participant_study) for example in examples]

    max_neighbors = max(25, len(examples))
    metric_norm = (metric or "cosine").lower()
    if metric_norm not in {"cosine", "l2", "euclidean"}:
        LOGGER.warning("[OPINION] Unsupported metric '%s'; falling back to cosine.", metric)
        metric_norm = "cosine"
    neighbor_metric = "cosine" if metric_norm == "cosine" else "euclidean"
    neighbors = NearestNeighbors(
        n_neighbors=min(max_neighbors, len(examples)),
        metric=neighbor_metric,
        algorithm="brute",
    )
    neighbors.fit(matrix)

    LOGGER.info(
        "[OPINION] Built %s index for study=%s participants=%d",
        feature_space.upper(),
        spec.key,
        len(examples),
    )

    return OpinionIndex(
        feature_space=feature_space,
        metric=metric_norm,
        matrix=matrix,
        vectorizer=vectorizer,
        embeds=embeds,
        targets_after=targets_after,
        targets_before=targets_before,
        participant_keys=participant_keys,
        neighbors=neighbors,
    )


def _transform_documents(
    *,
    index: OpinionIndex,
    documents: Sequence[str],
) -> Any:
    """Transform ``documents`` into the feature space of ``index``."""

    if index.feature_space == "tfidf":
        if index.vectorizer is None:
            raise RuntimeError("TF-IDF vectoriser missing from index.")
        return index.vectorizer.transform(documents).astype(np.float32)
    if index.feature_space == "word2vec":
        if index.embeds is None:
            raise RuntimeError("Word2Vec builder missing from index.")
        return index.embeds.transform(list(documents))
    raise ValueError(f"Unsupported feature space '{index.feature_space}'.")


def _similarity_from_distances(distances: np.ndarray, *, metric: str) -> np.ndarray:
    """Convert neighbour distances into similarity weights."""

    metric_norm = (metric or "cosine").lower()
    distances = np.asarray(distances, dtype=np.float32)
    if metric_norm == "cosine":
        similarities = 1.0 - distances
        return np.clip(similarities, 0.0, None)
    weights = 1.0 / (distances + 1e-6)
    return np.clip(weights, 0.0, None)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Return the weighted mean of ``values`` with fallback for zero weights."""

    total = float(weights.sum())
    if total <= 1e-8:
        return float(values.mean()) if values.size else float("nan")
    return float(np.dot(values, weights) / total)


def predict_post_indices(
    *,
    index: OpinionIndex,
    eval_examples: Sequence[OpinionExample],
    k_values: Sequence[int],
) -> Dict[str, Any]:
    """Return predictions and aggregate metrics for ``eval_examples``."""

    if not eval_examples:
        return {
            "rows": [],
            "per_k_predictions": {int(k): [] for k in k_values},
        }

    unique_k = sorted({int(k) for k in k_values if int(k) > 0})
    max_k = min(max(unique_k), len(index.participant_keys))

    documents = [example.document for example in eval_examples]
    matrix_eval = _transform_documents(index=index, documents=documents)

    neighbour_distances, neighbour_indices = index.neighbors.kneighbors(
        matrix_eval,
        n_neighbors=max_k,
    )

    per_k_predictions: Dict[int, List[float]] = {k: [] for k in unique_k}
    per_k_change_predictions: Dict[int, List[float]] = {k: [] for k in unique_k}
    rows: List[Dict[str, Any]] = []

    for row_idx, example in enumerate(eval_examples):
        distances = neighbour_distances[row_idx]
        indices = neighbour_indices[row_idx]
        similarities = _similarity_from_distances(distances, metric=index.metric)

        record_after: Dict[int, float] = {}
        record_change: Dict[int, float] = {}
        for k in unique_k:
            top_indices = indices[:k]
            top_weights = similarities[:k]
            top_targets_after = index.targets_after[top_indices]
            top_targets_before = index.targets_before[top_indices]
            top_changes = top_targets_after - top_targets_before

            predicted_change = _weighted_mean(top_changes, top_weights)
            anchored_prediction = float(example.before) + predicted_change

            per_k_predictions[k].append(anchored_prediction)
            per_k_change_predictions[k].append(predicted_change)
            record_after[k] = float(anchored_prediction)
            record_change[k] = float(predicted_change)

        rows.append(
            {
                "participant_id": example.participant_id,
                "participant_study": example.participant_study,
                "issue": example.issue,
                "session_id": example.session_id,
                "before_index": example.before,
                "after_index": example.after,
                "predictions_by_k": record_after,
                "predicted_change_by_k": record_change,
            }
        )

    return {
        "rows": rows,
        "per_k_predictions": per_k_predictions,
        "per_k_change_predictions": per_k_change_predictions,
    }


def _summary_metrics(
    *,
    predictions: Dict[int, List[float]],
    eval_examples: Sequence[OpinionExample],
) -> Dict[int, Dict[str, float]]:
    """Return MAE, RMSE, and R^2 metrics for each ``k``."""

    truth_after = np.asarray([example.after for example in eval_examples], dtype=np.float32)
    truth_before = np.asarray([example.before for example in eval_examples], dtype=np.float32)
    change_truth = truth_after - truth_before

    metrics: Dict[int, Dict[str, float]] = {}
    for k, preds in predictions.items():
        preds_arr = np.asarray(preds, dtype=np.float32)
        change_pred = preds_arr - truth_before
        mae = float(mean_absolute_error(truth_after, preds_arr))
        rmse = float(math.sqrt(mean_squared_error(truth_after, preds_arr)))
        r2 = float(r2_score(truth_after, preds_arr))
        change_mae = float(mean_absolute_error(change_truth, change_pred))

        metrics[int(k)] = {
            "mae_after": mae,
            "rmse_after": rmse,
            "r2_after": r2,
            "mae_change": change_mae,
        }
    return metrics


def _baseline_metrics(eval_examples: Sequence[OpinionExample]) -> Dict[str, float]:
    """Return baseline error metrics for opinion prediction."""

    truth_after = np.asarray([example.after for example in eval_examples], dtype=np.float32)
    truth_before = np.asarray([example.before for example in eval_examples], dtype=np.float32)
    change_truth = truth_after - truth_before

    baseline_mean = float(truth_after.mean()) if truth_after.size else float("nan")
    mae_mean = float(mean_absolute_error(truth_after, np.full_like(truth_after, baseline_mean)))
    rmse_mean = float(math.sqrt(mean_squared_error(truth_after, np.full_like(truth_after, baseline_mean))))

    mae_no_change = float(mean_absolute_error(truth_after, truth_before))
    change_zero = np.zeros_like(change_truth)
    mae_change_zero = float(mean_absolute_error(change_truth, change_zero))

    return {
        "mae_global_mean_after": mae_mean,
        "rmse_global_mean_after": rmse_mean,
        "mae_using_before": mae_no_change,
        "mae_change_zero": mae_change_zero,
        "global_mean_after": baseline_mean,
    }


def _plot_metric(
    *,
    metrics_by_k: Dict[int, Dict[str, float]],
    metric_key: str,
    output_path: Path,
) -> None:
    """Save a line plot of ``metric_key`` vs. ``k`` if matplotlib is available."""

    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping %s plot (matplotlib not installed).", metric_key)
        return

    if not metrics_by_k:
        LOGGER.warning("[OPINION] Skipping %s plot (no metrics).", metric_key)
        return

    sorted_items = sorted(metrics_by_k.items())
    ks = [item[0] for item in sorted_items]
    values = [item[1][metric_key] for item in sorted_items]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, values, marker="o")
    plt.title(f"{metric_key} vs k")
    plt.xlabel("k")
    plt.ylabel(metric_key)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_change_heatmap(
    *,
    actual_changes: Sequence[float],
    predicted_changes: Sequence[float],
    output_path: Path,
) -> None:
    """Render a 2D histogram comparing actual vs. predicted opinion shifts."""

    if plt is None:  # pragma: no cover - optional dependency
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (matplotlib not installed).")
        return

    if not actual_changes or not predicted_changes:
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (no valid predictions).")
        return

    actual = np.asarray(actual_changes, dtype=np.float32)
    predicted = np.asarray(predicted_changes, dtype=np.float32)
    if actual.size == 0 or predicted.size == 0:
        LOGGER.warning("[OPINION] Skipping opinion-change heatmap (empty arrays).")
        return

    min_val = float(min(actual.min(), predicted.min()))
    max_val = float(max(actual.max(), predicted.max()))
    if math.isclose(min_val, max_val):
        span = 0.1 if math.isfinite(min_val) else 1.0
        min_val -= span
        max_val += span
    else:
        extent = max(abs(min_val), abs(max_val))
        if not math.isfinite(extent) or extent <= 1e-6:
            extent = 1.0
        min_val, max_val = -extent, extent

    bins = 40
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.5, 4.5))
    hist = plt.hist2d(
        actual,
        predicted,
        bins=bins,
        range=[[min_val, max_val], [min_val, max_val]],
        cmap="magma",
        cmin=1,
    )
    plt.colorbar(hist[3], label="Participants")
    plt.plot([min_val, max_val], [min_val, max_val], color="cyan", linestyle="--", linewidth=1.0)
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axhline(0.0, color="grey", linestyle=":", linewidth=0.8)
    plt.axvline(0.0, color="grey", linestyle=":", linewidth=0.8)
    plt.xlabel("Actual opinion change (post - pre)")
    plt.ylabel("Predicted opinion change")
    plt.title("Predicted vs. actual opinion change")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _write_outputs(
    *,
    args,
    spec: OpinionSpec,
    index: OpinionIndex,
    rows: Sequence[Dict[str, Any]],
    metrics_by_k: Dict[int, Dict[str, float]],
    baseline: Dict[str, float],
    best_k: int,
    outputs_root: Path,
) -> None:
    """Persist per-example predictions, metrics, and plots."""

    feature_space = index.feature_space
    study_dir = outputs_root / spec.key
    study_dir.mkdir(parents=True, exist_ok=True)

    actual_changes: List[float] = []
    predicted_changes: List[float] = []
    for row in rows:
        prediction = row["predictions_by_k"].get(best_k)
        if prediction is None:
            continue
        before = float(row["before_index"])
        actual_changes.append(float(row["after_index"]) - before)
        predicted_change = row.get("predicted_change_by_k", {}).get(best_k)
        if predicted_change is None:
            predicted_change = float(prediction) - before
        predicted_changes.append(float(predicted_change))

    predictions_path = study_dir / f"opinion_knn_{spec.key}_{EVAL_SPLIT}.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for row in rows:
            change_dict = {
                str(k): float(v) for k, v in row.get("predicted_change_by_k", {}).items()
            }
            serializable = {
                **row,
                "predictions_by_k": {str(k): float(v) for k, v in row["predictions_by_k"].items()},
                "predicted_change_by_k": change_dict,
            }
            handle.write(json.dumps(serializable, ensure_ascii=False) + "\n")

    reports_dir = resolve_reports_dir(Path(args.out_dir)) / "knn" / feature_space / "opinion"
    reports_dir.mkdir(parents=True, exist_ok=True)
    mae_plot = reports_dir / f"mae_{spec.key}.png"
    r2_plot = reports_dir / f"r2_{spec.key}.png"
    _plot_metric(metrics_by_k=metrics_by_k, metric_key="mae_after", output_path=mae_plot)
    _plot_metric(metrics_by_k=metrics_by_k, metric_key="r2_after", output_path=r2_plot)
    heatmap_path = reports_dir / f"change_heatmap_{spec.key}.png"
    _plot_change_heatmap(
        actual_changes=actual_changes,
        predicted_changes=predicted_changes,
        output_path=heatmap_path,
    )

    metrics_path = study_dir / f"opinion_knn_{spec.key}_{EVAL_SPLIT}_metrics.json"
    serializable_metrics = {
        "model": "knn_opinion",
        "feature_space": feature_space,
        "dataset": args.dataset or DEFAULT_DATASET_SOURCE,
        "study": spec.key,
        "issue": spec.issue,
        "label": spec.label,
        "split": EVAL_SPLIT,
        "n_participants": len(rows),
        "metrics_by_k": {
            str(k): {
                "mae_after": float(values["mae_after"]),
                "rmse_after": float(values["rmse_after"]),
                "r2_after": float(values["r2_after"]),
                "mae_change": float(values["mae_change"]),
            }
            for k, values in metrics_by_k.items()
        },
        "baseline": baseline,
        "best_k": int(best_k),
        "best_metrics": metrics_by_k.get(int(best_k), {}),
        "plots": {
            "mae_vs_k": str(mae_plot),
            "r2_vs_k": str(r2_plot),
            "change_heatmap": str(heatmap_path),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(serializable_metrics, handle, ensure_ascii=False, indent=2)

    LOGGER.info(
        "[OPINION] Wrote predictions=%s metrics=%s",
        predictions_path,
        metrics_path,
    )


def run_opinion_eval(args) -> None:  # pylint: disable=too-many-locals
    """Execute the post-study opinion index evaluation."""

    os_env = os.environ
    os_env.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os_env.setdefault("HF_HOME", args.cache_dir)

    dataset_source = args.dataset or DEFAULT_DATASET_SOURCE
    dataset = load_dataset_source(dataset_source, args.cache_dir)

    if args.opinion_studies:
        requested = [
            token.strip() for token in args.opinion_studies.split(",") if token.strip()
        ]
    else:
        requested = [spec.key for spec in DEFAULT_SPECS]

    specs = [find_spec(token) for token in requested]

    extra_fields = [
        token.strip()
        for token in (args.knn_text_fields or "").split(",")
        if token.strip()
    ]

    k_values = parse_k_values(args.knn_k, args.knn_k_sweep)
    LOGGER.info("[OPINION] Evaluating k values: %s", k_values)

    feature_space = str(getattr(args, "feature_space", "tfidf")).lower()
    word2vec_cfg = None
    if feature_space == "word2vec":
        word2vec_cfg = Word2VecConfig(
            vector_size=int(args.word2vec_size),
            window=int(getattr(args, "word2vec_window", Word2VecConfig().window)),
            min_count=int(getattr(args, "word2vec_min_count", Word2VecConfig().min_count)),
            epochs=int(getattr(args, "word2vec_epochs", Word2VecConfig().epochs)),
            model_dir=Path(args.word2vec_model_dir or Word2VecConfig().model_dir) / "opinion",
            seed=int(getattr(args, "knn_seed", Word2VecConfig().seed)),
            workers=int(getattr(args, "word2vec_workers", Word2VecConfig().workers)),
        )

    outputs_root = Path(args.out_dir) / "opinion" / feature_space
    outputs_root.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        LOGGER.info("[OPINION] Study=%s (%s)", spec.key, spec.label)
        train_examples = collect_examples(
            dataset[TRAIN_SPLIT],
            spec=spec,
            extra_fields=extra_fields,
            max_examples=int(getattr(args, "knn_max_train", 0) or 0),
            seed=int(getattr(args, "knn_seed", 42)),
        )
        if not train_examples:
            LOGGER.warning("[OPINION] No training examples found for study=%s", spec.key)
            continue

        index = build_index(
            examples=train_examples,
            feature_space=feature_space,
            spec=spec,
            seed=int(getattr(args, "knn_seed", 42)),
            metric=str(getattr(args, "knn_metric", "cosine")),
            word2vec_config=word2vec_cfg,
        )

        eval_examples = collect_examples(
            dataset[EVAL_SPLIT],
            spec=spec,
            extra_fields=extra_fields,
            max_examples=int(getattr(args, "eval_max", 0) or 0),
            seed=int(getattr(args, "knn_seed", 42)),
        )
        if not eval_examples:
            LOGGER.warning("[OPINION] No evaluation examples found for study=%s", spec.key)
            continue

        predictions_bundle = predict_post_indices(
            index=index,
            eval_examples=eval_examples,
            k_values=k_values,
        )
        rows = predictions_bundle["rows"]
        metrics_by_k = _summary_metrics(
            predictions=predictions_bundle["per_k_predictions"],
            eval_examples=eval_examples,
        )

        if metrics_by_k:
            best_k, _ = max(metrics_by_k.items(), key=lambda item: item[1]["r2_after"])
        else:
            best_k = k_values[0]

        baseline = _baseline_metrics(eval_examples)
        _write_outputs(
            args=args,
            spec=spec,
            index=index,
            rows=rows,
            metrics_by_k=metrics_by_k,
            baseline=baseline,
            best_k=best_k,
            outputs_root=outputs_root,
        )

        LOGGER.info(
            "[OPINION][DONE] study=%s participants=%d best_k=%d r2=%.4f mae=%.4f",
            spec.key,
            len(rows),
            best_k,
            float(metrics_by_k.get(best_k, {}).get("r2_after", float("nan"))),
            float(metrics_by_k.get(best_k, {}).get("mae_after", float("nan"))),
        )


__all__ = ["run_opinion_eval", "OpinionSpec", "DEFAULT_SPECS"]

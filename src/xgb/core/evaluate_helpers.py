#!/usr/bin/env python
"""Helpers extracted from the XGBoost evaluation loop.

These utilities keep :mod:`xgb.core.evaluate` lean and avoid duplication.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence
from common.pipeline.utils import make_placeholder_metrics

# No evaluation-metric helpers imported here; keep this module focused on I/O and plumbing.
from .evaluation_records import compute_group_keys
from .evaluation_types import IssueMetrics
from .model import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    XGBoostBoosterParams,
    XGBoostSlateModel,
    XGBoostTrainConfig,
    fit_xgboost_model,
    load_xgboost_model,
    save_xgboost_model,
)
from .vectorizers import build_word2vec_config_from_args
from .utils import ensure_directory, get_logger


logger = get_logger("xgb.eval.helpers")


def split_tokens(raw: Optional[str]) -> List[str]:
    """Split a comma-delimited string into trimmed tokens.

    :param raw: Raw comma-separated string provided via CLI flags.
    :returns: Sequence of non-empty tokens.
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def log_training_validation_metrics(
    issue_slug: str,
    history: Mapping[str, Mapping[str, Sequence[object]]] | None,
) -> None:
    """Emit the latest validation metrics captured during booster training."""

    if not history:
        return
    summary_bits: List[str] = []
    for dataset_name, metrics_map in sorted(history.items()):
        if not isinstance(metrics_map, Mapping):
            continue
        if "valid" not in dataset_name.lower():
            continue
        metric_parts: List[str] = []
        for metric_name, values in sorted(metrics_map.items()):
            if not isinstance(values, Sequence) or not values:
                continue
            last_value = values[-1]
            try:
                formatted = f"{float(last_value):.4f}"
            except (TypeError, ValueError):
                continue
            metric_parts.append(f"{metric_name}={formatted}")
        if metric_parts:
            summary_bits.append(f"{dataset_name}: " + ", ".join(metric_parts))
    if summary_bits:
        logger.info(
            "[XGBoost][Train][Validation] issue=%s %s",
            issue_slug,
            " | ".join(summary_bits),
        )


def load_or_train_model(
    args,
    issue_slug: str,
    train_ds,
    eval_ds,
    extra_fields: Sequence[str],
) -> XGBoostSlateModel:
    """Return a trained or loaded XGBoost model for the requested issue.

    :param args: Parsed CLI namespace containing training options.
    :param issue_slug: Normalised issue identifier.
    :param train_ds: Training dataset split.
    :param eval_ds: Evaluation dataset split used for history capture.
    :param extra_fields: Extra text fields passed to the feature builder.
    :returns: :class:`XGBoostSlateModel` ready for evaluation.
    :raises ValueError: If neither ``--fit-model`` nor ``--load-model`` is specified.
    """

    if args.fit_model:
        logger.info("[XGBoost] Training model for issue=%s", issue_slug)
        booster_params = XGBoostBoosterParams.create(
            learning_rate=args.xgb_learning_rate,
            max_depth=args.xgb_max_depth,
            n_estimators=args.xgb_n_estimators,
            subsample=args.xgb_subsample,
            colsample_bytree=args.xgb_colsample_bytree,
            tree_method=args.xgb_tree_method,
            reg_lambda=args.xgb_reg_lambda,
            reg_alpha=args.xgb_reg_alpha,
        )
        word2vec_model_dir = args.word2vec_model_dir
        if word2vec_model_dir:
            word2vec_model_dir = str(Path(word2vec_model_dir) / issue_slug)
        train_config = XGBoostTrainConfig.create(
            max_train=args.max_train,
            seed=args.seed,
            max_features=args.max_features if args.max_features else None,
            vectorizer_kind=getattr(args, "text_vectorizer", "tfidf"),
            tfidf=TfidfConfig(max_features=args.max_features if args.max_features else None),
            word2vec=build_word2vec_config_from_args(args, model_dir=word2vec_model_dir),
            sentence_transformer=SentenceTransformerVectorizerConfig(
                model_name=args.sentence_transformer_model,
                device=args.sentence_transformer_device,
                batch_size=args.sentence_transformer_batch_size,
                normalize=args.sentence_transformer_normalize,
            ),
            booster=booster_params,
        )
        model = fit_xgboost_model(
            train_ds,
            config=train_config,
            extra_fields=extra_fields,
            eval_ds=eval_ds,
            collect_history=True,
        )
        if args.save_model:
            save_xgboost_model(model, Path(args.save_model) / issue_slug)
        return model

    if args.load_model:
        logger.info("[XGBoost] Loading model for issue=%s", issue_slug)
        return load_xgboost_model(Path(args.load_model) / issue_slug)

    raise ValueError("Set either --fit_model or --load_model to obtain an XGBoost model.")


def write_outputs(
    args,
    issue_slug: str,
    metrics: IssueMetrics,
    predictions: List[Dict[str, Any]],
) -> None:
    """Persist metrics and predictions for a single issue evaluation."""

    out_dir = Path(args.out_dir) / issue_slug
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace outputs.")
    ensure_directory(out_dir)
    if metrics.curve_metrics:
        curve_path = out_dir / f"xgb_curves_{issue_slug}.json"
        with open(curve_path, "w", encoding="utf-8") as handle:
            json.dump(metrics.curve_metrics, handle, indent=2)
        metrics.curve_metrics_path = str(curve_path)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(metrics), handle, indent=2)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")


def write_skip_metrics(
    *,
    args,
    issue_slug: str,
    metadata: Mapping[str, Any],
) -> None:
    """Persist a minimal metrics.json payload indicating a skipped evaluation.

    :param args: Parsed CLI namespace containing output options.
    :param issue_slug: Issue identifier used for output directory naming.
    :param metadata: Mapping that must contain keys ``participant_studies``,
        ``dataset_source``, ``extra_fields``, and ``reason``.
    """
    out_dir = Path(args.out_dir) / issue_slug
    ensure_directory(out_dir)
    skipped_payload = make_placeholder_metrics(
        issue_slug,
        list(metadata.get("participant_studies", [])),
        extra_fields=list(metadata.get("extra_fields", [])),
        skip_reason=metadata.get("reason", "Skipped"),
    )
    skipped_payload["dataset_source"] = metadata.get("dataset_source")
    skipped_payload["timestamp"] = time.time()
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(skipped_payload, handle, indent=2)


def bootstrap_settings() -> tuple[int, int]:
    """Return (replicates, seed) parsed from environment with defaults."""
    try:
        replicates = int(os.environ.get("XGB_BOOTSTRAP_REPLICATES", "500"))
    except ValueError:
        replicates = 500
    try:
        bootstrap_seed = int(os.environ.get("XGB_BOOTSTRAP_SEED", "2024"))
    except ValueError:
        bootstrap_seed = 2024
    return replicates, bootstrap_seed


def group_keys_with_fallback(eval_ds, n_records: int) -> List[str]:
    """Return group keys, falling back to per-row identifiers when needed."""
    try:
        return compute_group_keys(eval_ds, n_records)
    except (TypeError, AttributeError):  # pragma: no cover - defensive fallback
        return [f"row::{i}" for i in range(n_records)]


def baseline_top_index(metrics: IssueMetrics) -> Optional[int]:
    """Extract the baseline top-index from metrics when available."""
    payload = metrics.baseline_most_frequent_gold_index or {}
    if isinstance(payload, Mapping):
        try:
            value = payload.get("top_index")
            return int(value) if value is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive
            return None
    return None


def attach_uncertainty(metrics: IssueMetrics, uncertainty: Mapping[str, Any] | None) -> None:
    """Attach uncertainty summaries to ``metrics`` when present."""
    if not uncertainty or not isinstance(uncertainty, Mapping):
        return
    model_uncertainty = uncertainty.get("model")
    if isinstance(model_uncertainty, Mapping):
        # Assign defensively in case metrics does not expose optional fields.
        try:
            metrics.accuracy_ci_95 = model_uncertainty.get("ci95")  # type: ignore[assignment]
            metrics.accuracy_uncertainty = uncertainty  # type: ignore[assignment]
        except AttributeError:  # pragma: no cover - attribute differences
            return


__all__ = [
    "attach_uncertainty",
    "baseline_top_index",
    "bootstrap_settings",
    "group_keys_with_fallback",
    "load_or_train_model",
    "log_training_validation_metrics",
    "write_skip_metrics",
    "split_tokens",
    "write_outputs",
]

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

"""Evaluation loop and metrics for the XGBoost slate baseline."""

from __future__ import annotations

# pylint: disable=duplicate-code,too-many-lines

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common.evaluation.utils import compose_issue_slug, prepare_dataset, safe_div
from common.prompts.docs import merge_default_extra_fields

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    filter_dataset_for_issue,
    filter_split_for_participant_studies,
    issues_in_dataset,
    load_dataset_source,
)
from .model import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
    XGBoostBoosterParams,
    XGBoostSlateModel,
    XGBoostTrainConfig,
    fit_xgboost_model,
    load_xgboost_model,
    save_xgboost_model,
)
from .evaluation_metrics import (
    accuracy_curve_from_records,
    bootstrap_uncertainty,
    curve_metrics_for_split,
    curve_metrics_from_training_history,
    model_params,
    records_to_predictions,
    summarise_outcomes,
    summarise_records,
)
from .evaluation_probabilities import candidate_probabilities, probability_context
from .evaluation_records import (
    collect_prediction_records,
    compute_group_keys,
    evaluate_single_example,
    group_key_for_example,
)
from .evaluation_types import (
    EvaluationConfig,
    IssueEvaluationContext,
    IssueMetrics,
    PredictionOutcome,
)
from .utils import ensure_directory, get_logger

# Backwards-compatible aliases for legacy imports (tests, downstream scripts).
_candidate_probabilities = candidate_probabilities
_probability_context = probability_context
_collect_prediction_records = collect_prediction_records
_evaluate_single_example = evaluate_single_example
_group_key_for_example = group_key_for_example
_compute_group_keys = compute_group_keys
_records_to_predictions = records_to_predictions
_accuracy_curve_from_records = accuracy_curve_from_records
_curve_metrics_from_training_history = curve_metrics_from_training_history
_curve_metrics_for_split = curve_metrics_for_split
_summarise_outcomes = summarise_outcomes
_summarise_records = summarise_records
_bootstrap_uncertainty = bootstrap_uncertainty
_model_params = model_params

logger = get_logger("xgb.eval")


def _log_training_validation_metrics(
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


def _split_tokens(raw: Optional[str]) -> List[str]:
    """
    Split a comma-delimited string into trimmed tokens.

    :param raw: Raw comma-separated string provided via CLI flags.
    :type raw: Optional[str]
    :returns: Sequence of non-empty tokens.
    :rtype: List[str]
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def run_eval(args) -> None:
    """
    Evaluate the XGBoost baseline across the requested issues.

    :param args: Parsed CLI arguments produced via :func:`xgb.cli.build_parser`.
    :type args: argparse.Namespace
    """
    dataset_source, base_ds, available_issues = prepare_dataset(
        dataset=getattr(args, "dataset", None),
        default_source=DEFAULT_DATASET_SOURCE,
        cache_dir=args.cache_dir,
        loader=load_dataset_source,
        issue_lookup=issues_in_dataset,
    )

    if args.issues:
        requested = [token.strip() for token in args.issues.split(",") if token.strip()]
        issues = requested if requested else available_issues
    else:
        issues = available_issues

    joint_study_tokens = _split_tokens(getattr(args, "participant_studies", ""))
    train_study_tokens = (
        _split_tokens(getattr(args, "train_participant_studies", "")) or joint_study_tokens
    )
    eval_study_tokens = (
        _split_tokens(getattr(args, "eval_participant_studies", "")) or joint_study_tokens
    )

    extra_fields = merge_default_extra_fields(_split_tokens(args.extra_text_fields))

    for issue in issues:
        _evaluate_issue(
            args,
            issue,
            base_ds,
            context=IssueEvaluationContext(
                dataset_source=dataset_source,
                extra_fields=tuple(extra_fields),
                train_study_tokens=tuple(train_study_tokens),
                eval_study_tokens=tuple(eval_study_tokens),
            ),
        )


# pylint: disable=too-many-arguments,too-many-locals
def _evaluate_issue(
    args,
    issue: str,
    base_ds,
    *,
    context: IssueEvaluationContext,
) -> None:
    """Evaluate a single issue for the XGBoost baseline and persist outputs.

    :param args: Parsed CLI namespace controlling training/evaluation options.
    :param issue: Issue label (human-readable) requested for evaluation.
    :param base_ds: Loaded dataset dictionary containing train/eval splits.
    :param context: Static context describing dataset metadata and participant filters.
    :type context: IssueEvaluationContext
    """

    train_tokens = [token for token in context.train_study_tokens if token]
    eval_tokens = [token for token in context.eval_study_tokens if token]
    issue_slug = compose_issue_slug(issue, eval_tokens)

    logger.info(
        "[XGBoost] Evaluating issue=%s train_studies=%s eval_studies=%s",
        issue_slug,
        ",".join(train_tokens) or "all",
        ",".join(eval_tokens) or "all",
    )

    issue_dataset = filter_dataset_for_issue(base_ds, issue)
    train_ds = filter_split_for_participant_studies(issue_dataset[TRAIN_SPLIT], train_tokens)
    eval_ds = filter_split_for_participant_studies(issue_dataset[EVAL_SPLIT], eval_tokens)

    train_rows = len(train_ds)
    eval_rows = len(eval_ds)

    # Emit dataset sizes up-front for sweep visibility
    logger.info(
        "[XGBoost] issue=%s train_rows=%d eval_rows=%d train_studies=%s eval_studies=%s",
        issue_slug,
        train_rows,
        eval_rows,
        ",".join(train_tokens) or "all",
        ",".join(eval_tokens) or "all",
    )

    if train_rows == 0 or eval_rows == 0:
        logger.warning(
            "[XGBoost] Skipping issue=%s (train_rows=%d eval_rows=%d) after participant "
            "study filters (train=%s eval=%s).",
            issue_slug,
            train_rows,
            eval_rows,
            ",".join(train_tokens) or "all",
            ",".join(eval_tokens) or "all",
        )
        # Emit a minimal metrics.json so downstream reuse/caching can detect a skip.
        out_dir = Path(args.out_dir) / issue_slug
        ensure_directory(out_dir)
        skipped_payload = {
            "issue": issue_slug,
            "participant_studies": list(eval_tokens),
            "dataset_source": context.dataset_source,
            "evaluated": 0,
            "correct": 0,
            "accuracy": 0.0,
            "known_candidate_hits": 0,
            "known_candidate_total": 0,
            "coverage": 0.0,
            "eligible": 0,
            "timestamp": time.time(),
            "extra_fields": list(context.extra_fields),
            "skipped": True,
            "skip_reason": "No train/eval rows after filters",
        }
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(skipped_payload, handle, indent=2)
        return

    model = _load_or_train_model(
        args,
        issue_slug,
        train_ds,
        eval_ds,
        context.extra_fields,
    )
    _log_training_validation_metrics(issue_slug, getattr(model, "training_history", None))

    eval_config = EvaluationConfig(
        dataset_source=context.dataset_source,
        extra_fields=tuple(context.extra_fields),
        eval_max=args.eval_max,
        participant_studies=tuple(eval_tokens),
    )
    metrics, predictions, eval_curve = evaluate_issue(
        model=model,
        eval_ds=eval_ds,
        issue_slug=issue_slug,
        config=eval_config,
    )
    history_bundle = curve_metrics_from_training_history(model.training_history)
    if history_bundle is None:
        curve_bundle: Dict[str, Any] = {
            "axis_label": "Evaluated examples",
            "y_label": "Cumulative accuracy",
            "eval": eval_curve,
        }
        train_curve = curve_metrics_for_split(
            model=model,
            dataset=train_ds,
            extra_fields=tuple(context.extra_fields),
        )
        if train_curve.get("n_examples"):
            curve_bundle["train"] = train_curve
        metrics.curve_metrics = curve_bundle
    else:
        metrics.curve_metrics = history_bundle
    _write_outputs(args, issue_slug, metrics, predictions)
    # Diagnostic: compare eligible-only and known-candidate accuracy slices.
    try:
        known_total = int(metrics.known_candidate_total)
        known_hits = int(metrics.known_candidate_hits)
    except (TypeError, ValueError):  # pragma: no cover - defensive cast
        known_total = metrics.known_candidate_total
        known_hits = metrics.known_candidate_hits
    known_accuracy = safe_div(known_hits, known_total) if known_total else 0.0
    logger.info(
        "[XGBoost][Diag] eligible=%d known_total=%d known_hits=%d "
        "known_accuracy=%.3f eligible_accuracy=%.3f",
        metrics.eligible,
        metrics.known_candidate_total,
        metrics.known_candidate_hits,
        known_accuracy,
        metrics.accuracy_eligible,
    )
    logger.info(
        "[XGBoost][Validation] issue=%s accuracy=%.3f eligible_accuracy=%.3f "
        "known_accuracy=%.3f coverage=%.3f evaluated=%d",
        issue_slug,
        metrics.accuracy,
        metrics.accuracy_eligible,
        known_accuracy,
        metrics.coverage,
        metrics.evaluated,
    )
    logger.info(
        "[XGBoost] Issue=%s accuracy=%.3f coverage=%.3f evaluated=%d",
        issue_slug,
        metrics.accuracy,
        metrics.coverage,
        metrics.evaluated,
    )


def _load_or_train_model(
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
        booster_params = XGBoostBoosterParams(
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
        train_config = XGBoostTrainConfig(
            max_train=args.max_train,
            seed=args.seed,
            max_features=args.max_features if args.max_features else None,
            vectorizer_kind=getattr(args, "text_vectorizer", "tfidf"),
            tfidf=TfidfConfig(max_features=args.max_features if args.max_features else None),
            word2vec=Word2VecVectorizerConfig(
                vector_size=args.word2vec_size,
                window=args.word2vec_window,
                min_count=args.word2vec_min_count,
                epochs=args.word2vec_epochs,
                workers=args.word2vec_workers,
                seed=args.seed,
                model_dir=word2vec_model_dir,
            ),
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


def _write_outputs(
    args,
    issue_slug: str,
    metrics: IssueMetrics,
    predictions: List[Dict[str, Any]],
) -> None:
    """Persist metrics and predictions for a single issue evaluation.

    :param args: Parsed CLI namespace controlling output directory handling.
    :param issue_slug: Issue identifier appended to output paths.
    :param metrics: Summary metrics produced by :func:`evaluate_issue`.
    :param predictions: Per-example prediction dictionaries to serialise.
    :raises FileExistsError: If the output directory exists and ``--overwrite`` is not set.
    """

    out_dir = Path(args.out_dir) / issue_slug
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} already exists. Use --overwrite to replace outputs."
        )
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


# pylint: disable=too-many-locals
def evaluate_issue(
    *,
    model: XGBoostSlateModel,
    eval_ds,
    issue_slug: str,
    config: EvaluationConfig,
) -> tuple[IssueMetrics, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Evaluate a trained XGBoost model on the provided evaluation split.

    :param model: Trained model bundle used to score slate options.
    :type model: XGBoostSlateModel
    :param eval_ds: Dataset split representing evaluation rows.
    :type eval_ds: datasets.Dataset or sequence-like
    :param issue_slug: Slug-ified representation of the issue being evaluated.
    :type issue_slug: str
    :param config: Evaluation configuration bundle (dataset info, extra fields, limits).
    :type config: EvaluationConfig
    :returns: Pair of summary metrics and per-example prediction details.
    :rtype: tuple[IssueMetrics, List[Dict[str, Any]]]
    """

    records = collect_prediction_records(model, eval_ds, config)
    metrics = summarise_records(records, config, issue_slug, model)
    predictions = records_to_predictions(records, issue_slug)
    curve_payload = accuracy_curve_from_records(records)
    # Compute participant-bootstrap CIs for eligible-only accuracy.
    try:
        group_keys = compute_group_keys(eval_ds, len(records))
    except (TypeError, AttributeError):  # pragma: no cover - defensive fallback
        group_keys = [f"row::{i}" for i in range(len(records))]
    baseline_index = None
    payload = metrics.baseline_most_frequent_gold_index or {}
    if isinstance(payload, Mapping):
        baseline_index = payload.get("top_index")
    try:
        replicates = int(os.environ.get("XGB_BOOTSTRAP_REPLICATES", "500"))
    except ValueError:
        replicates = 500
    try:
        bootstrap_seed = int(os.environ.get("XGB_BOOTSTRAP_SEED", "2024"))
    except ValueError:
        bootstrap_seed = 2024
    uncertainty = bootstrap_uncertainty(
        records=records,
        group_keys=group_keys,
        baseline_index=baseline_index,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    if uncertainty and isinstance(uncertainty, Mapping):
        model_uncertainty = uncertainty.get("model")
        if isinstance(model_uncertainty, Mapping):
            metrics.accuracy_ci_95 = model_uncertainty.get("ci95")  # type: ignore[assignment]
            metrics.accuracy_uncertainty = uncertainty  # type: ignore[assignment]
    return metrics, predictions, curve_payload


__all__ = [
    "EvaluationConfig",
    "IssueMetrics",
    "PredictionOutcome",
    "evaluate_issue",
    "run_eval",
    "safe_div",
]

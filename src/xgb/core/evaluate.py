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

from typing import Any, Dict, List, Sequence
from dataclasses import dataclass

from common.evaluation.utils import (
    compose_issue_slug,
    prepare_dataset_from_args,
    safe_div,
)
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
from .model import XGBoostSlateModel
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
from .utils import get_logger
from .evaluate_helpers import (
    attach_uncertainty as _attach_uncertainty,
    baseline_top_index as _baseline_top_index,
    bootstrap_settings as _bootstrap_settings,
    group_keys_with_fallback as _group_keys_with_fallback,
    load_or_train_model as _load_or_train_model,
    log_training_validation_metrics as _log_training_validation_metrics,
    split_tokens as _split_tokens,
    write_outputs as _write_outputs,
)
from .evaluate_helpers import write_skip_metrics as _write_skip_metrics

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


    # helpers moved to xgb.core.evaluate_helpers


# Expose a monkeypatch-friendly dataset resolver under this module.
def prepare_dataset(args, *, default_source, loader, issue_lookup):
    """Compatibility wrapper delegating to prepare_dataset_from_args.

    Tests patch xgb.evaluate.prepare_dataset; defining this indirection here
    lets those patches take effect without changing the underlying behaviour.
    """
    return prepare_dataset_from_args(
        args,
        default_source=default_source,
        loader=loader,
        issue_lookup=issue_lookup,
    )


def run_eval(args) -> None:
    """
    Evaluate the XGBoost baseline across the requested issues.

    :param args: Parsed CLI arguments produced via :func:`xgb.cli.build_parser`.
    :type args: argparse.Namespace
    """
    dataset_source, base_ds, available_issues = prepare_dataset(
        args=args,
        default_source=DEFAULT_DATASET_SOURCE,
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


@dataclass(frozen=True)
class _FinalizeContext:
    issue_slug: str
    model: XGBoostSlateModel
    train_ds: Any
    extra_fields: Sequence[str]


def _finalize_and_write_outputs(
    args,
    ctx: _FinalizeContext,
    eval_result: tuple[IssueMetrics, List[Dict[str, Any]], Dict[str, Any]],
) -> IssueMetrics:
    """Attach curve metrics, persist outputs, and return metrics.

    This helper reduces local variables in the caller by handling
    curve construction and output persistence in one place.
    """
    metrics, predictions, eval_curve = eval_result
    history_bundle = curve_metrics_from_training_history(ctx.model.training_history)
    if history_bundle is None:
        curve_bundle: Dict[str, Any] = {
            "axis_label": "Evaluated examples",
            "y_label": "Cumulative accuracy",
            "eval": eval_curve,
        }
        train_curve = curve_metrics_for_split(
            model=ctx.model,
            dataset=ctx.train_ds,
            extra_fields=tuple(ctx.extra_fields),
        )
        if train_curve.get("n_examples"):
            curve_bundle["train"] = train_curve
        metrics.curve_metrics = curve_bundle
    else:
        metrics.curve_metrics = history_bundle
    _write_outputs(args, ctx.issue_slug, metrics, predictions)
    return metrics


def _log_eval_diagnostics(issue_slug: str, metrics: IssueMetrics) -> None:
    """Emit diagnostic summary lines derived from final metrics."""
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

    # Emit dataset sizes up-front for sweep visibility
    logger.info(
        "[XGBoost] issue=%s train_rows=%d eval_rows=%d train_studies=%s eval_studies=%s",
        issue_slug,
        len(train_ds),
        len(eval_ds),
        ",".join(train_tokens) or "all",
        ",".join(eval_tokens) or "all",
    )

    if len(train_ds) == 0 or len(eval_ds) == 0:
        logger.warning(
            "[XGBoost] Skipping issue=%s (train_rows=%d eval_rows=%d) after participant "
            "study filters (train=%s eval=%s).",
            issue_slug,
            len(train_ds),
            len(eval_ds),
            ",".join(train_tokens) or "all",
            ",".join(eval_tokens) or "all",
        )
        # Emit a minimal metrics.json so downstream reuse/caching can detect a skip.
        _write_skip_metrics(
            args=args,
            issue_slug=issue_slug,
            metadata={
                "participant_studies": list(eval_tokens),
                "dataset_source": context.dataset_source,
                "extra_fields": list(context.extra_fields),
                "reason": "No train/eval rows after filters",
            },
        )
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
    metrics = _finalize_and_write_outputs(
        args,
        _FinalizeContext(
            issue_slug=issue_slug,
            model=model,
            train_ds=train_ds,
            extra_fields=tuple(context.extra_fields),
        ),
        evaluate_issue(
            model=model,
            eval_ds=eval_ds,
            issue_slug=issue_slug,
            config=eval_config,
        ),
    )
    _log_eval_diagnostics(issue_slug, metrics)


    # load_or_train_model moved to xgb.core.evaluate_helpers


    # write_outputs moved to xgb.core.evaluate_helpers


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
    group_keys = _group_keys_with_fallback(eval_ds, len(records))
    baseline_index = _baseline_top_index(metrics)
    replicates, bootstrap_seed = _bootstrap_settings()
    uncertainty = bootstrap_uncertainty(
        records=records,
        group_keys=group_keys,
        baseline_index=baseline_index,
        replicates=replicates,
        seed=bootstrap_seed,
    )
    _attach_uncertainty(metrics, uncertainty)
    return metrics, predictions, curve_payload


    # bootstrap_settings moved to xgb.core.evaluate_helpers


    # group_keys_with_fallback moved to xgb.core.evaluate_helpers


    # baseline_top_index moved to xgb.core.evaluate_helpers


    # attach_uncertainty moved to xgb.core.evaluate_helpers


__all__ = [
    "EvaluationConfig",
    "IssueMetrics",
    "PredictionOutcome",
    "evaluate_issue",
    "run_eval",
    "safe_div",
]

"""Evaluation loop and metrics for the XGBoost slate baseline."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

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
from .model import (
    XGBoostBoosterParams,
    XGBoostSlateModel,
    XGBoostTrainConfig,
    fit_xgboost_model,
    load_xgboost_model,
    predict_among_slate,
    save_xgboost_model,
)
from .utils import canon_video_id, ensure_directory, get_logger

logger = get_logger("xgb.eval")


def safe_div(numerator: float, denominator: float) -> float:
    """
    Return the division result guarding against a zero denominator.

    :param numerator: Value forming the numerator.
    :type numerator: float
    :param denominator: Value forming the denominator.
    :type denominator: float
    :returns: ``numerator / denominator`` or ``0.0`` when the denominator is zero.
    :rtype: float
    """

    return numerator / denominator if denominator else 0.0


# pylint: disable=too-many-instance-attributes
@dataclass
class IssueMetrics:
    """Container describing evaluation metrics for a single issue."""

    issue: str
    dataset_source: str
    evaluated: int
    correct: int
    accuracy: float
    known_candidate_hits: int
    known_candidate_total: int
    coverage: float
    avg_probability: float
    timestamp: float
    extra_fields: Sequence[str]
    xgboost_params: Dict[str, Any]


@dataclass(frozen=True)
class EvaluationConfig:
    """
    Configuration bundle shared across evaluation helpers.

    :ivar dataset_source: Identifier for the dataset source (path or HF id).
    :vartype dataset_source: str
    :ivar extra_fields: Additional column names appended to prompt documents.
    :vartype extra_fields: Sequence[str]
    :ivar eval_max: Optional cap on the number of evaluation rows (0 evaluates all).
    :vartype eval_max: int
    """

    dataset_source: str
    extra_fields: Sequence[str]
    eval_max: int


# pylint: disable=too-many-instance-attributes
@dataclass(frozen=True)
class PredictionOutcome:
    """
    Result bundle for a single evaluation example.

    :ivar prediction_index: 1-based index of the chosen slate option (``None`` when unknown).
    :vartype prediction_index: Optional[int]
    :ivar predicted_id: Video identifier selected by the model.
    :vartype predicted_id: str
    :ivar gold_video_id: Ground-truth video identifier.
    :vartype gold_video_id: str
    :ivar candidate_probs: Mapping of slate positions to probabilities.
    :vartype candidate_probs: Dict[int, float]
    :ivar best_probability: Probability associated with the predicted option.
    :vartype best_probability: float
    :ivar known_candidate_seen:
        Flag indicating whether any slate ids were present in the probability map.
    :vartype known_candidate_seen: bool
    :ivar known_candidate_hit:
        Flag indicating the predicted option matched the ground-truth id and was known.
    :vartype known_candidate_hit: bool
    :ivar record_probability:
        Flag indicating whether ``best_probability`` should be included in aggregates.
    :vartype record_probability: bool
    :ivar correct: ``True`` when the predicted option matches the gold id.
    :vartype correct: bool
    """

    prediction_index: Optional[int]
    predicted_id: str
    gold_video_id: str
    candidate_probs: Dict[int, float]
    best_probability: float
    known_candidate_seen: bool
    known_candidate_hit: bool
    record_probability: bool
    correct: bool


@dataclass(frozen=True)
class ProbabilityContext:
    """Aggregated probability metadata for a slate prediction."""

    best_probability: float
    record_probability: bool
    known_candidate_hit: bool


@dataclass(frozen=True)
class OutcomeSummary:
    """Aggregated metrics derived from prediction outcomes."""

    evaluated: int
    correct: int
    known_hits: int
    known_total: int
    avg_probability: float


def run_eval(args) -> None:
    """
    Evaluate the XGBoost baseline across the requested issues.

    :param args: Parsed CLI arguments produced via :func:`xgb.cli.build_parser`.
    :type args: argparse.Namespace
    """

    os_env = os.environ
    os_env.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os_env.setdefault("HF_HOME", args.cache_dir)

    dataset_source = args.dataset or DEFAULT_DATASET_SOURCE
    base_ds = load_dataset_source(dataset_source, args.cache_dir)
    available_issues = issues_in_dataset(base_ds)

    if args.issues:
        requested = [token.strip() for token in args.issues.split(",") if token.strip()]
        issues = requested if requested else available_issues
    else:
        issues = available_issues

    extra_fields = [
        token.strip()
        for token in (args.extra_text_fields or "").split(",")
        if token.strip()
    ]

    for issue in issues:
        _evaluate_issue(args, issue, base_ds, dataset_source, extra_fields)


def _evaluate_issue(
    args,
    issue: str,
    base_ds,
    dataset_source: str,
    extra_fields: List[str],
) -> None:
    issue_slug = issue.replace(" ", "_") if issue and issue.strip() else "all"
    logger.info("[XGBoost] Evaluating issue=%s", issue_slug)
    ds = filter_dataset_for_issue(base_ds, issue)
    train_ds = ds[TRAIN_SPLIT]
    eval_ds = ds[EVAL_SPLIT]

    model = _load_or_train_model(args, issue_slug, train_ds, extra_fields)

    eval_config = EvaluationConfig(
        dataset_source=dataset_source,
        extra_fields=tuple(extra_fields),
        eval_max=args.eval_max,
    )
    metrics, predictions = evaluate_issue(
        model=model,
        eval_ds=eval_ds,
        issue_slug=issue_slug,
        config=eval_config,
    )

    _write_outputs(args, issue_slug, metrics, predictions)
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
    extra_fields: Sequence[str],
) -> XGBoostSlateModel:
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
        train_config = XGBoostTrainConfig(
            max_train=args.max_train,
            seed=args.seed,
            max_features=args.max_features if args.max_features else None,
            booster=booster_params,
        )
        model = fit_xgboost_model(
            train_ds,
            config=train_config,
            extra_fields=extra_fields,
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
    out_dir = Path(args.out_dir) / issue_slug
    if out_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"{out_dir} already exists. Use --overwrite to replace outputs."
        )
    ensure_directory(out_dir)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(asdict(metrics), handle, indent=2)
    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row) + "\n")


def evaluate_issue(
    *,
    model: XGBoostSlateModel,
    eval_ds,
    issue_slug: str,
    config: EvaluationConfig,
) -> tuple[IssueMetrics, List[Dict[str, Any]]]:
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

    records = _collect_prediction_records(model, eval_ds, config)
    metrics = _summarise_records(records, config, issue_slug, model)
    predictions = _records_to_predictions(records, issue_slug)
    return metrics, predictions


def _evaluate_single_example(
    *,
    model: XGBoostSlateModel,
    example: dict,
    extra_fields: Sequence[str],
) -> PredictionOutcome:
    """Return the prediction outcome for a single example."""
    prediction_idx, probability_map = predict_among_slate(
        model,
        example,
        extra_fields=extra_fields,
    )
    slate = extract_slate_items(example)
    gold_id = example.get(SOLUTION_COLUMN) or ""
    gold_id_canon = canon_video_id(gold_id)

    if prediction_idx is None and slate:
        prediction_idx = 1

    if prediction_idx is not None and 1 <= prediction_idx <= len(slate):
        predicted_id = slate[prediction_idx - 1][1]
    else:
        predicted_id = ""

    candidate_probs, known_candidates = _candidate_probabilities(slate, probability_map)
    probability_ctx = _probability_context(
        prediction_idx=prediction_idx,
        candidate_probs=candidate_probs,
        known_candidates=known_candidates,
        gold_id_canon=gold_id_canon,
    )

    predicted_id_canon = canon_video_id(predicted_id)
    correct = predicted_id_canon == gold_id_canon and bool(predicted_id_canon)

    return PredictionOutcome(
        prediction_index=prediction_idx,
        predicted_id=predicted_id,
        gold_video_id=gold_id,
        candidate_probs=candidate_probs,
        best_probability=probability_ctx.best_probability,
        known_candidate_seen=bool(known_candidates),
        known_candidate_hit=probability_ctx.known_candidate_hit,
        record_probability=probability_ctx.record_probability,
        correct=correct,
    )


def _collect_prediction_records(
    model: XGBoostSlateModel,
    eval_ds,
    config: EvaluationConfig,
) -> List[tuple[int, PredictionOutcome]]:
    """Return indexed prediction outcomes for the evaluation split."""
    records: List[tuple[int, PredictionOutcome]] = []
    for index, example in enumerate(eval_ds):
        if config.eval_max and len(records) >= config.eval_max:
            break
        outcome = _evaluate_single_example(
            model=model,
            example=example,
            extra_fields=config.extra_fields,
        )
        records.append((index, outcome))
    return records


def _summarise_records(
    records: List[tuple[int, PredictionOutcome]],
    config: EvaluationConfig,
    issue_slug: str,
    model: XGBoostSlateModel,
) -> IssueMetrics:
    """Aggregate prediction records into an :class:`IssueMetrics` summary."""
    summary = _summarise_outcomes(records)
    return IssueMetrics(
        issue=issue_slug,
        dataset_source=config.dataset_source,
        evaluated=summary.evaluated,
        correct=summary.correct,
        accuracy=safe_div(summary.correct, summary.evaluated),
        known_candidate_hits=summary.known_hits,
        known_candidate_total=summary.known_total,
        coverage=safe_div(summary.known_hits, summary.known_total),
        avg_probability=summary.avg_probability,
        timestamp=time.time(),
        extra_fields=tuple(config.extra_fields),
        xgboost_params=_model_params(model),
    )


def _records_to_predictions(
    records: List[tuple[int, PredictionOutcome]],
    issue_slug: str,
) -> List[Dict[str, Any]]:
    """Serialise prediction records into API-friendly dictionaries."""
    return [
        {
            "issue": issue_slug,
            "index": index,
            "prediction_index": outcome.prediction_index,
            "predicted_video_id": outcome.predicted_id,
            "gold_video_id": outcome.gold_video_id,
            "correct": outcome.correct,
            "probabilities": outcome.candidate_probs,
        }
        for index, outcome in records
    ]


def _candidate_probabilities(
    slate: Sequence[tuple[str, str]],
    probability_map: Dict[str, float],
) -> tuple[Dict[int, float], Dict[int, str]]:
    candidate_probs = {
        slate_idx + 1: probability_map.get(canon_video_id(candidate_id), 0.0)
        for slate_idx, (_, candidate_id) in enumerate(slate)
    }
    known_candidates = {
        slate_idx + 1: canon_video_id(candidate_id)
        for slate_idx, (_, candidate_id) in enumerate(slate)
        if canon_video_id(candidate_id) in probability_map
    }
    return candidate_probs, known_candidates


def _probability_context(
    *,
    prediction_idx: Optional[int],
    candidate_probs: Dict[int, float],
    known_candidates: Dict[int, str],
    gold_id_canon: str,
) -> ProbabilityContext:
    best_probability = (
        candidate_probs.get(prediction_idx, 0.0)
        if prediction_idx is not None
        else 0.0
    )
    record_probability = bool(prediction_idx and prediction_idx in known_candidates)
    known_candidate_hit = bool(
        record_probability
        and prediction_idx is not None
        and known_candidates[prediction_idx] == gold_id_canon
    )
    return ProbabilityContext(
        best_probability=best_probability,
        record_probability=record_probability,
        known_candidate_hit=known_candidate_hit,
    )


def _summarise_outcomes(
    records: List[tuple[int, PredictionOutcome]]
) -> OutcomeSummary:
    outcomes = [outcome for _, outcome in records]
    evaluated = len(outcomes)
    known_total = sum(outcome.known_candidate_seen for outcome in outcomes)
    known_hits = sum(outcome.known_candidate_hit for outcome in outcomes)
    probability_values = [
        outcome.best_probability
        for outcome in outcomes
        if outcome.record_probability
    ]
    avg_probability = float(np.mean(probability_values)) if probability_values else 0.0
    correct = sum(outcome.correct for outcome in outcomes)
    return OutcomeSummary(
        evaluated=evaluated,
        correct=correct,
        known_hits=known_hits,
        known_total=known_total,
        avg_probability=avg_probability,
    )


def _model_params(model: XGBoostSlateModel) -> Dict[str, Any]:
    """
    Return a serialisable view of relevant XGBoost parameters.

    :param model: Model bundle whose configuration should be summarised.
    :type model: XGBoostSlateModel
    :returns: Dictionary containing key training parameters.
    :rtype: Dict[str, Any]
    """

    params = model.booster.get_params()
    selected = {
        key: params.get(key)
        for key in [
            "objective",
            "eval_metric",
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "tree_method",
            "reg_lambda",
            "reg_alpha",
        ]
    }
    selected["extra_fields"] = list(model.extra_fields)
    selected["n_features"] = int(getattr(model.vectorizer, "max_features", 0) or 0)
    selected["n_classes"] = int(len(model.label_encoder.classes_))
    return selected


__all__ = ["EvaluationConfig", "IssueMetrics", "evaluate_issue", "run_eval", "safe_div"]

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

# pylint: disable=duplicate-code

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from common.eval_utils import compose_issue_slug, prepare_dataset, safe_div

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    SOLUTION_COLUMN,
    TRAIN_SPLIT,
    filter_dataset_for_issue,
    filter_dataset_for_participant_studies,
    issues_in_dataset,
    load_dataset_source,
)
from .features import extract_slate_items
from .model import (
    SentenceTransformerVectorizerConfig,
    TfidfConfig,
    Word2VecVectorizerConfig,
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


# pylint: disable=too-many-instance-attributes
@dataclass
class IssueMetrics:
    """Container describing evaluation metrics for a single issue."""

    issue: str
    participant_studies: Sequence[str]
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
    curve_metrics: Optional[Dict[str, Any]] = None
    curve_metrics_path: Optional[str] = None


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
    :ivar participant_studies: Tokenised participant study filters applied to the splits.
    :vartype participant_studies: Sequence[str]
    """

    dataset_source: str
    extra_fields: Sequence[str]
    eval_max: int
    participant_studies: Sequence[str]


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
    """
    Aggregated probability metadata for a slate prediction.

    :ivar best_probability: Probability assigned to the chosen candidate.
    :vartype best_probability: float
    :ivar record_probability: Flag indicating whether the probability should contribute
        to averages (only when the candidate was observed during training).
    :vartype record_probability: bool
    :ivar known_candidate_hit: Flag signalling that the predicted candidate matches
        the gold label and was seen during training.
    :vartype known_candidate_hit: bool
    """

    best_probability: float
    record_probability: bool
    known_candidate_hit: bool


@dataclass(frozen=True)
class OutcomeSummary:
    """
    Aggregated metrics derived from prediction outcomes.

    :ivar evaluated: Number of evaluation rows processed.
    :vartype evaluated: int
    :ivar correct: Count of correct slate selections.
    :vartype correct: int
    :ivar known_hits: Count of correct selections among candidates observed during training.
    :vartype known_hits: int
    :ivar known_total: Count of evaluations with at least one known candidate.
    :vartype known_total: int
    :ivar avg_probability: Mean probability assigned to known predictions.
    :vartype avg_probability: float
    """

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

    study_tokens = _split_tokens(getattr(args, "participant_studies", ""))

    extra_fields = [
        token.strip()
        for token in (args.extra_text_fields or "").split(",")
        if token.strip()
    ]

    for issue in issues:
        _evaluate_issue(
            args,
            issue,
            base_ds,
            dataset_source,
            extra_fields,
            study_tokens,
        )


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def _evaluate_issue(
    args,
    issue: str,
    base_ds,
    dataset_source: str,
    extra_fields: List[str],
    study_tokens: Sequence[str],
) -> None:
    """Evaluate a single issue for the XGBoost baseline and persist outputs.

    :param args: Parsed CLI namespace controlling training/evaluation options.
    :param issue: Issue label (human-readable) requested for evaluation.
    :param base_ds: Loaded dataset dictionary containing train/eval splits.
    :param dataset_source: String describing the data source (path or hub id).
    :param extra_fields: Additional text fields appended to the feature document.
    :param study_tokens: Participant study filters applied to train/eval splits.
    """

    tokens = [token for token in study_tokens if token]
    issue_slug = compose_issue_slug(issue, tokens)

    logger.info(
        "[XGBoost] Evaluating issue=%s participant_studies=%s",
        issue_slug,
        ",".join(tokens) or "all",
    )

    ds = filter_dataset_for_issue(base_ds, issue)
    if tokens:
        ds = filter_dataset_for_participant_studies(ds, tokens)
    train_ds = ds[TRAIN_SPLIT]
    eval_ds = ds[EVAL_SPLIT]

    train_rows = len(train_ds)
    eval_rows = len(eval_ds)

    if train_rows == 0 or eval_rows == 0:
        logger.warning(
            "[XGBoost] Skipping issue=%s (train_rows=%d eval_rows=%d) after participant "
            "study filter.",
            issue_slug,
            train_rows,
            eval_rows,
        )
        return

    model = _load_or_train_model(args, issue_slug, train_ds, extra_fields)

    eval_config = EvaluationConfig(
        dataset_source=dataset_source,
        extra_fields=tuple(extra_fields),
        eval_max=args.eval_max,
        participant_studies=tuple(tokens),
    )
    metrics, predictions, eval_curve = evaluate_issue(
        model=model,
        eval_ds=eval_ds,
        issue_slug=issue_slug,
        config=eval_config,
    )
    curve_bundle: Dict[str, Any] = {"eval": eval_curve}
    train_curve = _curve_metrics_for_split(
        model=model,
        dataset=train_ds,
        extra_fields=tuple(extra_fields),
    )
    if train_curve.get("n_examples"):
        curve_bundle["train"] = train_curve
    metrics.curve_metrics = curve_bundle
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
    """Return a trained or loaded XGBoost model for the requested issue.

    :param args: Parsed CLI namespace containing training options.
    :param issue_slug: Normalised issue identifier.
    :param train_ds: Training dataset split.
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

    records = _collect_prediction_records(model, eval_ds, config)
    metrics = _summarise_records(records, config, issue_slug, model)
    predictions = _records_to_predictions(records, issue_slug)
    curve_payload = _accuracy_curve_from_records(records)
    return metrics, predictions, curve_payload


def _evaluate_single_example(
    *,
    model: XGBoostSlateModel,
    example: dict,
    extra_fields: Sequence[str],
) -> PredictionOutcome:
    """
    Score a single interaction and package the outcome metadata.

    :param model: Trained slate model used for inference.
    :type model: XGBoostSlateModel
    :param example: Dataset row containing prompt text and slate candidates.
    :type example: dict
    :param extra_fields: Additional columns appended to the feature document.
    :type extra_fields: Sequence[str]
    :returns: Rich prediction bundle describing the model decision.
    :rtype: PredictionOutcome
    """
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
    """
    Collect indexed prediction outcomes for the evaluation split.

    :param model: Trained slate model used to score candidate lists.
    :type model: XGBoostSlateModel
    :param eval_ds: Iterable of dataset rows representing the evaluation split.
    :type eval_ds: datasets.Dataset | Sequence[dict]
    :param config: Evaluation configuration controlling maximum rows and extra fields.
    :type config: EvaluationConfig
    :returns: Ordered list mapping dataset indices to :class:`PredictionOutcome`.
    :rtype: List[tuple[int, PredictionOutcome]]
    """
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
    """
    Aggregate prediction records into an :class:`IssueMetrics` summary.

    :param records: Indexed prediction outcomes for the evaluation split.
    :type records: List[tuple[int, PredictionOutcome]]
    :param config: Evaluation configuration specifying dataset metadata.
    :type config: EvaluationConfig
    :param issue_slug: Slug identifying the evaluated issue.
    :type issue_slug: str
    :param model: Trained model bundle used to augment metrics with parameters.
    :type model: XGBoostSlateModel
    :returns: Metrics ready for serialisation to ``metrics.json``.
    :rtype: IssueMetrics
    """
    summary = _summarise_outcomes(records)
    return IssueMetrics(
        issue=issue_slug,
        participant_studies=tuple(config.participant_studies),
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
    """
    Serialise prediction outcomes into JSON-friendly dictionaries.

    :param records: Indexed prediction outcomes emitted by :func:`_collect_prediction_records`.
    :type records: List[tuple[int, PredictionOutcome]]
    :param issue_slug: Identifier describing the evaluated issue.
    :type issue_slug: str
    :returns: List of dictionaries mirroring the JSONL predictions format.
    :rtype: List[Dict[str, Any]]
    """
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


def _accuracy_curve_from_records(
    records: Sequence[tuple[int, PredictionOutcome]],
    *,
    target_points: int = 50,
) -> Dict[str, Any]:
    """
    Build cumulative accuracy checkpoints for plotting learning curves.

    :param records: Ordered prediction outcomes produced during evaluation.
    :type records: Sequence[tuple[int, PredictionOutcome]]
    :param target_points: Approximate number of checkpoints to retain.
    :type target_points: int
    :returns: Mapping containing the accuracy curve, total examples, and stride.
    :rtype: Dict[str, Any]
    """

    total = len(records)
    if total == 0:
        return {"accuracy_by_step": {}, "n_examples": 0, "stride": 0}
    target_points = max(1, target_points)
    stride = max(1, total // target_points)
    checkpoints: Dict[str, float] = {}
    correct = 0
    for idx, (_index, outcome) in enumerate(records, start=1):
        if outcome.correct:
            correct += 1
        if idx == total or idx % stride == 0:
            checkpoints[str(idx)] = safe_div(correct, idx)
    if str(total) not in checkpoints:
        checkpoints[str(total)] = safe_div(correct, total)
    return {
        "accuracy_by_step": checkpoints,
        "n_examples": total,
        "stride": stride,
    }


def _curve_metrics_for_split(
    model: XGBoostSlateModel,
    dataset,
    extra_fields: Sequence[str],
    *,
    target_points: int = 50,
) -> Dict[str, Any]:
    """
    Compute cumulative accuracy metrics for an arbitrary dataset split.

    :param model: Trained slate model used for inference.
    :type model: XGBoostSlateModel
    :param dataset: Iterable of dataset rows to evaluate.
    :type dataset: datasets.Dataset | Sequence[dict]
    :param extra_fields: Additional columns appended to the feature document.
    :type extra_fields: Sequence[str]
    :param target_points: Approximate number of checkpoints to retain.
    :type target_points: int
    :returns: Accuracy curve payload mirroring :func:`_accuracy_curve_from_records`.
    :rtype: Dict[str, Any]
    """

    config = EvaluationConfig(
        dataset_source="curve",
        extra_fields=tuple(extra_fields),
        eval_max=0,
        participant_studies=(),
    )
    records = _collect_prediction_records(model, dataset, config)
    return _accuracy_curve_from_records(records, target_points=target_points)


def _candidate_probabilities(
    slate: Sequence[tuple[str, str]],
    probability_map: Dict[str, float],
) -> tuple[Dict[int, float], Dict[int, str]]:
    """Map slate indices to predicted probabilities and known candidates.

    :param slate: Ordered sequence of slate options ``(title, video_id)``.
    :param probability_map: Mapping from canonical video id to predicted probability.
    :returns: Tuple of ``(candidate_probabilities, known_candidate_ids)`` keyed by 1-based index.
    :rtype: tuple[Dict[int, float], Dict[int, str]]
    """

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
    """Return context describing the probability associated with the prediction.

    :param prediction_idx: 1-based predicted index or ``None`` when absent.
    :param candidate_probs: Mapping from 1-based index to predicted probability.
    :param known_candidates: Mapping from 1-based index to canonical id when seen during training.
    :param gold_id_canon: Canonicalised gold video identifier.
    :returns: :class:`ProbabilityContext` describing probabilities and hits.
    :rtype: ProbabilityContext
    """

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
    """Aggregate prediction outcomes into summary counts.

    :param records: Sequence of ``(index, PredictionOutcome)`` tuples.
    :type records: List[tuple[int, PredictionOutcome]]
    :returns: :class:`OutcomeSummary` containing accuracy, coverage, and averages.
    :rtype: OutcomeSummary
    """

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
    if hasattr(model.vectorizer, "metadata"):
        vectorizer_meta = model.vectorizer.metadata()  # type: ignore[assignment]
        selected["vectorizer"] = vectorizer_meta
        selected["n_features"] = int(vectorizer_meta.get("dimension", 0))
    else:
        selected["n_features"] = int(getattr(model.vectorizer, "max_features", 0) or 0)
    selected["n_classes"] = int(len(model.label_encoder.classes_))
    return selected


__all__ = ["EvaluationConfig", "IssueMetrics", "evaluate_issue", "run_eval", "safe_div"]

"""Evaluation loop and metrics for the XGBoost slate baseline."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

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
from .model import XGBoostSlateModel, fit_xgboost_model, load_xgboost_model, predict_among_slate, save_xgboost_model
from .utils import ensure_directory, get_logger

logger = get_logger("xgboost.eval")


def safe_div(numerator: float, denominator: float) -> float:
    """Return the division result guarding against a zero denominator."""

    return numerator / denominator if denominator else 0.0


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


def run_eval(args) -> None:
    """Evaluate the XGBoost baseline across the requested issues."""

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
        issue_slug = issue.replace(" ", "_") if issue and issue.strip() else "all"
        logger.info("[XGBoost] Evaluating issue=%s", issue_slug)
        ds = filter_dataset_for_issue(base_ds, issue)
        train_ds = ds[TRAIN_SPLIT]
        eval_ds = ds[EVAL_SPLIT]

        if args.fit_model:
            logger.info("[XGBoost] Training model for issue=%s", issue_slug)
            model = fit_xgboost_model(
                train_ds,
                max_train=args.max_train,
                seed=args.seed,
                extra_fields=extra_fields,
                max_features=args.max_features,
                learning_rate=args.xgb_learning_rate,
                max_depth=args.xgb_max_depth,
                n_estimators=args.xgb_n_estimators,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                tree_method=args.xgb_tree_method,
                reg_lambda=args.xgb_reg_lambda,
                reg_alpha=args.xgb_reg_alpha,
            )
            if args.save_model:
                save_xgboost_model(model, Path(args.save_model) / issue_slug)
        elif args.load_model:
            logger.info("[XGBoost] Loading model for issue=%s", issue_slug)
            model = load_xgboost_model(Path(args.load_model) / issue_slug)
        else:
            raise ValueError("Set either --fit_model or --load_model to obtain an XGBoost model.")

        metrics, predictions = evaluate_issue(
            model=model,
            eval_ds=eval_ds,
            issue_slug=issue_slug,
            dataset_source=dataset_source,
            extra_fields=extra_fields,
            eval_max=args.eval_max,
        )

        out_dir = Path(args.out_dir) / issue_slug
        if out_dir.exists() and not args.overwrite:
            raise FileExistsError(f"{out_dir} already exists. Use --overwrite to replace outputs.")
        ensure_directory(out_dir)
        with open(out_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(asdict(metrics), handle, indent=2)
        with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as handle:
            for row in predictions:
                handle.write(json.dumps(row) + "\n")
        logger.info(
            "[XGBoost] Issue=%s accuracy=%.3f coverage=%.3f evaluated=%d",
            issue_slug,
            metrics.accuracy,
            metrics.coverage,
            metrics.evaluated,
        )


def evaluate_issue(
    *,
    model: XGBoostSlateModel,
    eval_ds,
    issue_slug: str,
    dataset_source: str,
    extra_fields: Sequence[str],
    eval_max: int,
) -> tuple[IssueMetrics, List[Dict[str, Any]]]:
    """Evaluate a trained XGBoost model on the provided evaluation split."""

    total = 0
    correct = 0
    known_candidate_hits = 0
    known_candidate_total = 0
    probability_accumulator: List[float] = []
    predictions: List[Dict[str, Any]] = []

    for index in range(len(eval_ds)):
        if eval_max and total >= eval_max:
            break
        example = eval_ds[index]
        total += 1
        prediction_idx, probability_map = predict_among_slate(
            model,
            example,
            extra_fields=extra_fields,
        )
        slate = extract_slate_items(example)
        gold_id = example.get(SOLUTION_COLUMN) or ""
        gold_id_canon = _canon_vid(gold_id)

        if prediction_idx is None and slate:
            prediction_idx = 1

        if prediction_idx is not None and 1 <= prediction_idx <= len(slate):
            predicted_id = slate[prediction_idx - 1][1]
        else:
            predicted_id = ""

        predicted_id_canon = _canon_vid(predicted_id)

        candidate_probs = {
            slate_idx + 1: probability_map.get(_canon_vid(candidate_id), 0.0)
            for slate_idx, (_, candidate_id) in enumerate(slate)
        }
        known_candidates = {
            slate_idx + 1: _canon_vid(candidate_id)
            for slate_idx, (_, candidate_id) in enumerate(slate)
            if _canon_vid(candidate_id) in probability_map
        }

        if known_candidates:
            known_candidate_total += 1
        best_probability = candidate_probs.get(prediction_idx, 0.0) if prediction_idx else 0.0
        if prediction_idx in known_candidates:
            probability_accumulator.append(best_probability)
            if known_candidates[prediction_idx] == gold_id_canon:
                known_candidate_hits += 1

        is_correct = predicted_id_canon == gold_id_canon and bool(predicted_id_canon)
        correct += int(is_correct)

        predictions.append(
            {
                "issue": issue_slug,
                "index": index,
                "prediction_index": prediction_idx,
                "predicted_video_id": predicted_id,
                "gold_video_id": gold_id,
                "correct": is_correct,
                "probabilities": candidate_probs,
            }
        )

    metrics = IssueMetrics(
        issue=issue_slug,
        dataset_source=dataset_source,
        evaluated=total,
        correct=correct,
        accuracy=safe_div(correct, total),
        known_candidate_hits=known_candidate_hits,
        known_candidate_total=known_candidate_total,
        coverage=safe_div(known_candidate_hits, known_candidate_total),
        avg_probability=float(np.mean(probability_accumulator)) if probability_accumulator else 0.0,
        timestamp=time.time(),
        extra_fields=tuple(extra_fields),
        xgboost_params=_model_params(model),
    )
    return metrics, predictions


_YTID_RE = re.compile(r"([A-Za-z0-9_-]{11})")


def _canon_vid(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    match = _YTID_RE.search(value)
    return match.group(1) if match else value.strip()


def _model_params(model: XGBoostSlateModel) -> Dict[str, Any]:
    """Return a serialisable view of relevant XGBoost parameters."""

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


__all__ = ["IssueMetrics", "evaluate_issue", "run_eval", "safe_div"]

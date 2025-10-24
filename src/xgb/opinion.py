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

"""Opinion-regression workflow powering the XGBoost baseline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from common.opinion import (
    DEFAULT_SPECS,
    OpinionExample,
    OpinionSpec,
    float_or_none,
    opinion_example_kwargs,
)

try:  # pragma: no cover - optional dependency
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:  # pragma: no cover - optional dependency
    TfidfVectorizer = None  # type: ignore[assignment]
    mean_absolute_error = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    r2_score = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from xgboost import XGBRegressor  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    XGBRegressor = None  # type: ignore[assignment]

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    load_dataset_source,
)
from .features import assemble_document
from .model import XGBoostBoosterParams
from .utils import get_logger

LOGGER = get_logger("xgb.opinion")


@dataclass(frozen=True)
class OpinionTrainConfig:
    """
    Training options shared across the opinion pipeline.

    :ivar max_participants: Optional cap on the number of participants (0 keeps all).
    :vartype max_participants: int
    :ivar seed: Random seed applied to participant sampling.
    :vartype seed: int
    :ivar max_features: Maximum TF-IDF features (``None`` allows full vocabulary).
    :vartype max_features: Optional[int]
    :ivar booster: Booster hyper-parameters reused for the regressor.
    :vartype booster: XGBoostBoosterParams
    """

    max_participants: int = 0
    seed: int = 42
    max_features: Optional[int] = None
    booster: XGBoostBoosterParams = field(default_factory=XGBoostBoosterParams)


@dataclass(frozen=True)
class OpinionEvalRequest:
    """
    Inputs required to execute the opinion regression workflow.

    :ivar dataset: Dataset identifier or path passed to :func:`load_dataset_source`.
    :vartype dataset: Optional[str]
    :ivar cache_dir: Optional cache directory for dataset loading.
    :vartype cache_dir: Optional[str]
    :ivar out_dir: Base directory receiving artefacts written by the stage.
    :vartype out_dir: Path
    :ivar feature_space: Identifier describing the feature representation (e.g. ``tfidf``).
    :vartype feature_space: str
    :ivar extra_fields: Additional prompt columns appended during document assembly.
    :vartype extra_fields: Sequence[str]
    :ivar train_config: Configuration applied during regressor training.
    :vartype train_config: OpinionTrainConfig
    :ivar overwrite: Flag controlling whether existing artefacts may be replaced.
    :vartype overwrite: bool
    """

    dataset: str | None
    cache_dir: str | None
    out_dir: Path
    feature_space: str
    extra_fields: Sequence[str]
    train_config: OpinionTrainConfig
    overwrite: bool = True


def _vectorizer_available() -> None:
    """
    Validate that optional dependencies required for opinion regression are installed.

    :raises ImportError: If either scikit-learn or XGBoost is unavailable.
    """

    if TfidfVectorizer is None or mean_absolute_error is None:  # pragma: no cover
        raise ImportError("Install scikit-learn to run the XGBoost opinion pipeline.")
    if XGBRegressor is None:  # pragma: no cover
        raise ImportError("Install xgboost to train the opinion regressor.")


# pylint: disable=too-many-locals
def collect_examples(
    dataset,
    *,
    spec: OpinionSpec,
    extra_fields: Sequence[str],
    max_participants: int,
    seed: int,
) -> List[OpinionExample]:
    """
    Collapse dataset rows down to one opinion example per participant.

    :param dataset: Dataset split providing raw interaction rows.
    :type dataset: datasets.Dataset | Sequence[dict]
    :param spec: Opinion study specification describing the target columns.
    :type spec: OpinionSpec
    :param extra_fields: Additional prompt columns appended to the document.
    :type extra_fields: Sequence[str]
    :param max_participants: Optional cap on the number of participants (0 keeps all).
    :type max_participants: int
    :param seed: Random seed used when subsampling participants.
    :type seed: int
    :returns: Participant-level examples combining prompts and opinion indices.
    :rtype: List[OpinionExample]
    """

    LOGGER.info(
        "[OPINION] Collapsing dataset for study=%s issue=%s rows=%d",
        spec.key,
        spec.issue,
        len(dataset),
    )
    per_participant: Dict[str, Tuple[OpinionExample, int]] = {}

    for raw in dataset:
        if raw.get("issue") != spec.issue or raw.get("participant_study") != spec.key:
            continue
        before = float_or_none(raw.get(spec.before_column))
        after = float_or_none(raw.get(spec.after_column))
        if before is None or after is None:
            continue
        document = assemble_document(raw, extra_fields)
        if not document:
            continue
        participant_id = str(raw.get("participant_id") or "")
        try:
            step_index = int(raw.get("step_index"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            step_index = -1
        base_kwargs = opinion_example_kwargs(
            participant_id=participant_id,
            participant_study=spec.key,
            issue=spec.issue,
            document=document,
            before=before,
            after=after,
        )
        example = OpinionExample(**base_kwargs)
        existing = per_participant.get(participant_id)
        if existing is None or step_index >= existing[1]:
            per_participant[participant_id] = (example, step_index)

    collapsed = [example for example, _ in per_participant.values()]
    LOGGER.info("[OPINION] Retained %d unique participants.", len(collapsed))

    if max_participants and len(collapsed) > max_participants:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(collapsed))[:max_participants]
        collapsed = [collapsed[i] for i in order]
        LOGGER.info("[OPINION] Sampled %d participants.", len(collapsed))
    return collapsed


def _fit_vectorizer(
    documents: Sequence[str],
    *,
    max_features: Optional[int],
) -> TfidfVectorizer:
    """
    Fit a TF-IDF vectoriser on the provided documents.

    :param documents: Corpus used to estimate TF-IDF statistics.
    :type documents: Sequence[str]
    :param max_features: Optional cap on the TF-IDF vocabulary size.
    :type max_features: Optional[int]
    :returns: Fitted scikit-learn TF-IDF vectoriser.
    :rtype: TfidfVectorizer
    """

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=1,
        max_features=max_features,
    )
    vectorizer.fit(documents)
    return vectorizer


def _train_regressor(
    *,
    features,
    targets: np.ndarray,
    config: OpinionTrainConfig,
    eval_features=None,
   eval_targets: np.ndarray | None = None,
) -> Tuple[XGBRegressor, Dict[str, Dict[str, List[float]]]]:
    """
    Train an :class:`xgboost.XGBRegressor` and capture optional evaluation metrics.

    :param features: Training feature matrix.
    :type features: Any
    :param targets: Training target vector.
    :type targets: numpy.ndarray
    :param config: Training configuration containing booster hyper-parameters.
    :type config: OpinionTrainConfig
    :param eval_features: Optional validation feature matrix.
    :type eval_features: Any, optional
    :param eval_targets: Optional validation target vector.
    :type eval_targets: Optional[numpy.ndarray]
    :returns: Tuple of ``(regressor, eval_history)`` where ``eval_history`` mirrors
        :meth:`xgboost.XGBRegressor.evals_result`.
    :rtype: Tuple[XGBRegressor, Dict[str, Dict[str, List[float]]]]
    """

    booster = config.booster
    regressor = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=booster.learning_rate,
        max_depth=booster.max_depth,
        n_estimators=booster.n_estimators,
        subsample=booster.subsample,
        colsample_bytree=booster.colsample_bytree,
        reg_lambda=booster.reg_lambda,
        reg_alpha=booster.reg_alpha,
        tree_method=booster.tree_method,
        n_jobs=-1,
        random_state=config.seed,
    )

    eval_set = []
    if eval_features is not None and eval_targets is not None:
        eval_set = [(features, targets), (eval_features, eval_targets)]
    if eval_set:
        regressor.set_params(eval_metric=["mae", "rmse"])
    regressor.fit(
        features,
        targets,
        eval_set=eval_set if eval_set else None,
        verbose=False,
    )
    history = regressor.evals_result() if eval_set else {}
    return regressor, history


def _direction_labels(delta: np.ndarray, *, tolerance: float = 1e-6) -> np.ndarray:
    """
    Return categorical labels capturing the direction of opinion change.

    :param delta: Opinion deltas computed as post-study minus pre-study indices.
    :type delta: numpy.ndarray
    :param tolerance: Magnitude threshold for treating changes as neutral.
    :type tolerance: float
    :returns: Array containing ``-1`` for decreases, ``0`` for neutral, ``1`` for increases.
    :rtype: numpy.ndarray
    """

    labels = np.zeros(delta.shape, dtype=np.int8)
    labels[delta > tolerance] = 1
    labels[delta < -tolerance] = -1
    return labels


def _baseline_metrics(
    *,
    before: np.ndarray,
    after: np.ndarray,
) -> Dict[str, float]:
    """
    Compute baseline metrics assuming the post-study index equals the pre-study value.

    :param before: Baseline opinion indices.
    :type before: numpy.ndarray
    :param after: Post-study opinion indices.
    :type after: numpy.ndarray
    :returns: Dictionary containing MAE, RMSE, and R² for the baseline predictor.
    :rtype: Dict[str, float]
    """

    mae = float(mean_absolute_error(after, before))
    rmse = float(math.sqrt(mean_squared_error(after, before)))
    r_squared = float(r2_score(after, before))
    change_truth = after - before
    direction_truth = _direction_labels(change_truth)
    direction_accuracy = (
        float(np.mean(direction_truth == 0))
        if direction_truth.size
        else float("nan")
    )
    return {
        "mae_before": mae,
        "rmse_before": rmse,
        "r2_before": r_squared,
        "direction_accuracy": direction_accuracy,
    }


def _model_metrics(
    *,
    predictions: np.ndarray,
    after: np.ndarray,
    before: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics comparing model predictions to ground truth.

    :param predictions: Predicted post-study opinion indices.
    :type predictions: numpy.ndarray
    :param after: Actual post-study opinion indices.
    :type after: numpy.ndarray
    :param before: Pre-study opinion indices (baseline values).
    :type before: numpy.ndarray
    :returns: Dictionary containing MAE, RMSE, and R² metrics.
    :rtype: Dict[str, float]
    """

    mae = float(mean_absolute_error(after, predictions))
    rmse = float(math.sqrt(mean_squared_error(after, predictions)))
    r_squared = float(r2_score(after, predictions))
    change_truth = after - before
    change_pred = predictions - before
    direction_truth = _direction_labels(change_truth)
    direction_pred = _direction_labels(change_pred)
    direction_accuracy = (
        float(np.mean(direction_truth == direction_pred))
        if direction_truth.size
        else float("nan")
    )
    eligible = int(direction_truth.size)
    return {
        "mae_after": mae,
        "rmse_after": rmse,
        "r2_after": r_squared,
        "direction_accuracy": direction_accuracy,
        "eligible": eligible,
    }


def _resolve_studies(tokens: Sequence[str]) -> List[OpinionSpec]:
    """
    Resolve CLI tokens into the corresponding opinion study specifications.

    :param tokens: Collection of study identifiers provided by the user.
    :type tokens: Sequence[str]
    :returns: Ordered list of matching :class:`OpinionSpec` definitions.
    :rtype: List[OpinionSpec]
    :raises ValueError: If an unknown study key is requested.
    """

    if not tokens:
        return list(DEFAULT_SPECS)
    resolved: List[OpinionSpec] = []
    valid = {spec.key.lower(): spec for spec in DEFAULT_SPECS}
    for token in tokens:
        normalised = token.strip().lower()
        if not normalised or normalised == "all":
            return list(DEFAULT_SPECS)
        try:
            resolved.append(valid[normalised])
        except KeyError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unknown opinion study '{token}'. Expected one of {sorted(valid)}."
            ) from exc
    return resolved


def _curve_metrics_from_history(eval_history: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Build per-round metric curves from the evaluation history, if available.

    :param eval_history: Training history captured by XGBoost.
    :type eval_history: Mapping[str, Any]
    :returns: Nested metric structure compatible with downstream reporting.
    :rtype: Optional[Dict[str, Any]]
    """

    if not eval_history:
        return None
    curve_metrics: Dict[str, Any] = {"metric": "mae"}
    dataset_map = {"validation_0": "train", "validation_1": "validation"}
    for history_key, label in dataset_map.items():
        history = eval_history.get(history_key, {})
        mae_sequence = history.get("mae", [])
        rmse_sequence = history.get("rmse", [])
        if not mae_sequence and not rmse_sequence:
            continue
        series_bundle: Dict[str, Dict[str, float]] = {}
        if mae_sequence:
            series_bundle["mae_by_round"] = {
                str(idx + 1): float(value) for idx, value in enumerate(mae_sequence)
            }
        if rmse_sequence:
            series_bundle["rmse_by_round"] = {
                str(idx + 1): float(value) for idx, value in enumerate(rmse_sequence)
            }
        curve_metrics[label] = series_bundle
    eval_mae_sequence = eval_history.get("validation_1", {}).get("mae", [])
    if eval_mae_sequence:
        best_round = min(
            range(len(eval_mae_sequence)),
            key=lambda idx: eval_mae_sequence[idx],
        )
        curve_metrics["best_round"] = int(best_round + 1)
        curve_metrics["best_mae"] = float(eval_mae_sequence[best_round])
    return curve_metrics


# pylint: disable=too-many-locals
def _evaluate_spec(
    *,
    dataset,
    spec: OpinionSpec,
    base_dir: Path,
    dataset_source: str,
    request: OpinionEvalRequest,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate a single opinion study and persist metrics plus predictions.

    :param dataset: Dataset dictionary containing train/eval splits.
    :type dataset: Mapping[str, Sequence[dict]]
    :param spec: Opinion study specification to evaluate.
    :type spec: OpinionSpec
    :param base_dir: Base output directory for study artefacts.
    :type base_dir: Path
    :param dataset_source: Human-readable dataset identifier for reporting.
    :type dataset_source: str
    :param request: Evaluation request describing feature space and configuration.
    :type request: OpinionEvalRequest
    :returns: Metrics payload summarising the evaluation, or ``None`` when skipped.
    :rtype: Optional[Dict[str, Any]]
    :raises FileExistsError: If outputs already exist and overwriting is disabled.
    """

    train_examples = collect_examples(
        dataset[TRAIN_SPLIT],
        spec=spec,
        extra_fields=request.extra_fields,
        max_participants=request.train_config.max_participants,
        seed=request.train_config.seed,
    )
    eval_examples = collect_examples(
        dataset[EVAL_SPLIT],
        spec=spec,
        extra_fields=request.extra_fields,
        max_participants=request.train_config.max_participants,
        seed=request.train_config.seed,
    )
    if not train_examples or not eval_examples:
        LOGGER.warning(
            "[OPINION] Skipping study=%s (train=%d eval=%d).",
            spec.key,
            len(train_examples),
            len(eval_examples),
        )
        return None

    vectorizer = _fit_vectorizer(
        [example.document for example in train_examples],
        max_features=request.train_config.max_features,
    )
    train_features = vectorizer.transform([ex.document for ex in train_examples])
    eval_features = vectorizer.transform([ex.document for ex in eval_examples])

    train_targets = np.array([ex.after for ex in train_examples], dtype=float)
    eval_targets = np.array([ex.after for ex in eval_examples], dtype=float)
    eval_before = np.array([ex.before for ex in eval_examples], dtype=float)
    regressor, eval_history = _train_regressor(
        features=train_features,
        targets=train_targets,
        config=request.train_config,
        eval_features=eval_features,
        eval_targets=eval_targets,
    )

    predictions = regressor.predict(eval_features)
    metrics = _model_metrics(predictions=predictions, after=eval_targets, before=eval_before)
    baseline = _baseline_metrics(
        before=eval_before,
        after=eval_targets,
    )

    study_dir = base_dir / spec.key
    if study_dir.exists() and not request.overwrite:
        raise FileExistsError(
            f"{study_dir} already exists. Use overwrite=True to replace outputs."
        )
    study_dir.mkdir(parents=True, exist_ok=True)

    curve_metrics = _curve_metrics_from_history(eval_history)

    payload: Dict[str, Any] = {
        "model": "xgb_opinion",
        "feature_space": request.feature_space,
        "dataset": dataset_source,
        "study": spec.key,
        "issue": spec.issue,
        "label": spec.label,
        "split": "validation",
        "n_participants": len(eval_examples),
        "train_participants": len(train_examples),
        "metrics": metrics,
        "baseline": baseline,
        "config": {
            "max_participants": request.train_config.max_participants,
            "max_features": request.train_config.max_features,
            "learning_rate": request.train_config.booster.learning_rate,
            "max_depth": request.train_config.booster.max_depth,
            "n_estimators": request.train_config.booster.n_estimators,
            "subsample": request.train_config.booster.subsample,
            "colsample_bytree": request.train_config.booster.colsample_bytree,
            "reg_lambda": request.train_config.booster.reg_lambda,
            "reg_alpha": request.train_config.booster.reg_alpha,
            "tree_method": request.train_config.booster.tree_method,
        },
    }
    if curve_metrics:
        payload["curve_metrics"] = curve_metrics
    eligible = metrics.get("eligible")
    if eligible is not None:
        payload["eligible"] = int(eligible)

    metrics_path = study_dir / f"opinion_xgb_{spec.key}_validation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    predictions_path = study_dir / f"opinion_xgb_{spec.key}_validation_predictions.jsonl"
    with open(predictions_path, "w", encoding="utf-8") as handle:
        for example, prediction in zip(eval_examples, predictions, strict=False):
            handle.write(
                json.dumps(
                    {
                        "participant_id": example.participant_id,
                        "study": spec.key,
                        "issue": spec.issue,
                        "before": example.before,
                        "after": example.after,
                        "prediction": float(prediction),
                        "error": float(abs(prediction - example.after)),
                    }
                )
                + "\n"
            )

    return payload


def run_opinion_eval(
    *,
    request: OpinionEvalRequest,
    studies: Sequence[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Execute the opinion regression workflow and persist artefacts under ``request.out_dir``.

    :param request: Evaluation request describing dataset and training configuration.
    :type request: OpinionEvalRequest
    :param studies: Optional subset of study keys to process (defaults to all).
    :type studies: Sequence[str] | None
    :returns: Mapping of study key to metric summary.
    :rtype: Dict[str, Dict[str, Any]]
    """

    _vectorizer_available()

    dataset_source = request.dataset or DEFAULT_DATASET_SOURCE
    dataset_bundle = load_dataset_source(dataset_source, request.cache_dir or "")
    selected_specs = _resolve_studies(studies or ())
    results: Dict[str, Dict[str, Any]] = {}

    base_dir = request.out_dir / request.feature_space
    base_dir.mkdir(parents=True, exist_ok=True)

    for spec in selected_specs:
        payload = _evaluate_spec(
            dataset=dataset_bundle,
            spec=spec,
            base_dir=base_dir,
            dataset_source=dataset_source,
            request=request,
        )
        if payload:
            results[spec.key] = payload
    return results


__all__ = [
    "OpinionSpec",
    "OpinionExample",
    "OpinionTrainConfig",
    "OpinionEvalRequest",
    "DEFAULT_SPECS",
    "collect_examples",
    "run_opinion_eval",
]

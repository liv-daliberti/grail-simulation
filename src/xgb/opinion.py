"""Opinion-index regression using XGBoost."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

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
class OpinionSpec:
    """Columns describing a single participant study."""

    key: str
    issue: str
    label: str
    before_column: str
    after_column: str


@dataclass
class OpinionExample:
    """Collapsed participant-level prompt and opinion indices."""

    participant_id: str
    participant_study: str
    issue: str
    document: str
    before: float
    after: float


@dataclass(frozen=True)
class OpinionTrainConfig:
    """Training options shared across the opinion pipeline."""

    max_participants: int = 0
    seed: int = 42
    max_features: Optional[int] = None
    booster: XGBoostBoosterParams = field(default_factory=XGBoostBoosterParams)


@dataclass(frozen=True)
class OpinionEvalRequest:
    """Inputs required to execute the opinion regression workflow."""

    dataset: str | None
    cache_dir: str | None
    out_dir: Path
    feature_space: str
    extra_fields: Sequence[str]
    train_config: OpinionTrainConfig
    overwrite: bool = True


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


def float_or_none(value: Any) -> Optional[float]:
    """Return ``value`` converted to ``float`` when possible."""

    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def _vectorizer_available() -> None:
    """Ensure optional scikit-learn and XGBoost dependencies are installed."""

    if TfidfVectorizer is None or mean_absolute_error is None:  # pragma: no cover
        raise ImportError("Install scikit-learn to run the XGBoost opinion pipeline.")
    if XGBRegressor is None:  # pragma: no cover
        raise ImportError("Install xgboost to train the opinion regressor.")


def collect_examples(
    dataset,
    *,
    spec: OpinionSpec,
    extra_fields: Sequence[str],
    max_participants: int,
    seed: int,
) -> List[OpinionExample]:
    """Collapse rows down to one entry per participant."""

    LOGGER.info(
        "[OPINION] Collapsing dataset for study=%s issue=%s rows=%d",
        spec.key,
        spec.issue,
        len(dataset),
    )
    per_participant: Dict[str, OpinionExample] = {}

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
        per_participant[participant_id] = OpinionExample(
            participant_id=participant_id,
            participant_study=spec.key,
            issue=spec.issue,
            document=document,
            before=before,
            after=after,
        )

    collapsed = list(per_participant.values())
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
    """Return TF-IDF vectoriser fitted on ``documents``."""

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
) -> XGBRegressor:
    """Return a fitted :class:`xgboost.XGBRegressor`."""

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
    regressor.fit(features, targets)
    return regressor


def _baseline_metrics(
    *,
    before: np.ndarray,
    after: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE/RMSE/R² when predicting ``after`` using ``before`` directly."""

    mae = float(mean_absolute_error(after, before))
    rmse = float(math.sqrt(mean_squared_error(after, before)))
    r2 = float(r2_score(after, before))
    return {
        "mae_before": mae,
        "rmse_before": rmse,
        "r2_before": r2,
    }


def _model_metrics(
    *,
    predictions: np.ndarray,
    after: np.ndarray,
) -> Dict[str, float]:
    """Compute MAE/RMSE/R² comparing model ``predictions`` to ``after`` targets."""

    mae = float(mean_absolute_error(after, predictions))
    rmse = float(math.sqrt(mean_squared_error(after, predictions)))
    r2 = float(r2_score(after, predictions))
    return {
        "mae_after": mae,
        "rmse_after": rmse,
        "r2_after": r2,
    }


def _resolve_studies(tokens: Sequence[str]) -> List[OpinionSpec]:
    """Return opinion study specifications matching ``tokens``."""

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


def _evaluate_spec(
    *,
    dataset,
    spec: OpinionSpec,
    base_dir: Path,
    dataset_source: str,
    request: OpinionEvalRequest,
) -> Optional[Dict[str, Any]]:
    """Evaluate a single study specification and persist outputs."""

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
    regressor = _train_regressor(
        features=train_features,
        targets=train_targets,
        config=request.train_config,
    )

    eval_targets = np.array([ex.after for ex in eval_examples], dtype=float)
    predictions = regressor.predict(eval_features)
    metrics = _model_metrics(predictions=predictions, after=eval_targets)
    baseline = _baseline_metrics(
        before=np.array([ex.before for ex in eval_examples], dtype=float),
        after=eval_targets,
    )

    study_dir = base_dir / spec.key
    if study_dir.exists() and not request.overwrite:
        raise FileExistsError(
            f"{study_dir} already exists. Use overwrite=True to replace outputs."
        )
    study_dir.mkdir(parents=True, exist_ok=True)

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

    :returns: Mapping of study key to metric summary.
    """

    _vectorizer_available()

    dataset_source = request.dataset or DEFAULT_DATASET_SOURCE
    ds = load_dataset_source(dataset_source, request.cache_dir or "")
    selected_specs = _resolve_studies(studies or ())
    results: Dict[str, Dict[str, Any]] = {}

    base_dir = request.out_dir / request.feature_space
    base_dir.mkdir(parents=True, exist_ok=True)

    for spec in selected_specs:
        payload = _evaluate_spec(
            dataset=ds,
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

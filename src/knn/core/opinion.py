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

"""Opinion-regression utilities supporting the KNN baselines."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from common.opinion import (
    DEFAULT_SPECS,
    OpinionSpec,
    exclude_eval_participants,
    ensure_train_examples,
    log_participant_counts,
)
from common.text.embeddings import (
    SentenceTransformerConfig,
    sentence_transformer_config_from_args,
)

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    load_dataset_source,
)
from .evaluate import parse_k_values
from .features import Word2VecConfig
from .opinion_data import (
    _extra_fields_from_args,
    _resolve_requested_specs,
    collect_examples,
)
from .opinion_index import build_index
from .opinion_models import (
    OpinionEmbeddingConfigs,
    OpinionEvaluationContext,
    OpinionExample,
    OpinionIndex,
    __all__ as _OPINION_MODEL_EXPORTS,
)
from .opinion_outputs import (
    _OutputContext,
    _OutputPayload,
    _write_outputs,
)
from .opinion_predictions import (
    _PredictionResults,
    _baseline_metrics,
    _curve_payload,
    _summary_metrics,
    predict_post_indices,
)

LOGGER = logging.getLogger("knn.opinion")

def _resolve_word2vec_config(args: Any, feature_space: str) -> Optional[Word2VecConfig]:
    """
    Return the Word2Vec configuration when enabled via ``feature_space``.

    :param args: CLI arguments containing optional overrides.
    :type args: Any
    :param feature_space: Requested feature space identifier.
    :type feature_space: str
    :returns: Configured :class:`~knn.core.features.Word2VecConfig` or ``None``.
    :rtype: Optional[Word2VecConfig]
    """
    if feature_space != "word2vec":
        return None
    defaults = Word2VecConfig()
    model_override = getattr(args, "word2vec_model_dir", None)
    model_dir = (
        Path(model_override)
        if model_override
        else Path("models/knn/opinions/word2vec_models")
    )
    return Word2VecConfig(
        vector_size=int(getattr(args, "word2vec_size", defaults.vector_size)),
        window=int(getattr(args, "word2vec_window", defaults.window)),
        min_count=int(getattr(args, "word2vec_min_count", defaults.min_count)),
        epochs=int(getattr(args, "word2vec_epochs", defaults.epochs)),
        model_dir=model_dir,
        seed=int(getattr(args, "knn_seed", defaults.seed)),
        workers=int(getattr(args, "word2vec_workers", defaults.workers)),
    )


def _resolve_sentence_config(
    args: Any,
    feature_space: str,
) -> Optional[SentenceTransformerConfig]:
    """
    Return the SentenceTransformer configuration when requested.

    :param args: CLI arguments containing optional overrides.
    :type args: Any
    :param feature_space: Requested feature space identifier.
    :type feature_space: str
    :returns: Configured :class:`SentenceTransformerConfig` or ``None``.
    :rtype: Optional[SentenceTransformerConfig]
    """
    if feature_space != "sentence_transformer":
        return None
    return sentence_transformer_config_from_args(args)


def _training_curve_metrics(
    index: OpinionIndex,
    train_examples: Sequence[OpinionExample],
    k_values: Sequence[int],
) -> Optional[Dict[str, Any]]:
    """
    Compute curve metrics for the training split, if available.

    :param index: Fitted opinion KNN index.
    :type index: OpinionIndex
    :param train_examples: Training split used to build the index.
    :type train_examples: Sequence[OpinionExample]
    :param k_values: Sequence of ``k`` values evaluated for the report.
    :type k_values: Sequence[int]
    :returns: Optional curve payload summarising the training split.
    :rtype: Optional[Dict[str, Any]]
    """
    if not train_examples:
        return None
    predictions_bundle = predict_post_indices(
        index=index,
        eval_examples=train_examples,
        k_values=k_values,
        exclude_self=True,
    )
    metrics_by_k = _summary_metrics(
        predictions=predictions_bundle["per_k_predictions"],
        eval_examples=train_examples,
        rows=predictions_bundle["rows"],
    )
    return _curve_payload(metrics_by_k, n_examples=len(train_examples))


def _evaluation_predictions(
    index: OpinionIndex,
    eval_examples: Sequence[OpinionExample],
    k_values: Sequence[int],
) -> _PredictionResults:
    """
    Return per-example predictions and summary metrics for evaluation examples.

    :param index: Fitted opinion KNN index.
    :type index: OpinionIndex
    :param eval_examples: Evaluation participants to score.
    :type eval_examples: Sequence[OpinionExample]
    :param k_values: Sequence of ``k`` values evaluated for the report.
    :type k_values: Sequence[int]
    :returns: Row-level predictions paired with aggregate metrics.
    :rtype: _PredictionResults
    """
    predictions_bundle = predict_post_indices(
        index=index,
        eval_examples=eval_examples,
        k_values=k_values,
    )
    rows = predictions_bundle["rows"]
    metrics_by_k = _summary_metrics(
        predictions=predictions_bundle["per_k_predictions"],
        eval_examples=eval_examples,
        rows=rows,
    )
    return _PredictionResults(rows=rows, metrics_by_k=metrics_by_k)


def _curve_metrics_bundle(
    metrics_by_k: Dict[int, Dict[str, float]],
    eval_examples: Sequence[OpinionExample],
    train_curve_metrics: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    Compose the optional train/eval curve payload persisted with metrics.

    :param metrics_by_k: Mapping from each ``k`` to opinion metrics.
    :type metrics_by_k: Dict[int, Dict[str, float]]
    :param eval_examples: Evaluation examples referenced by the metrics.
    :type eval_examples: Sequence[OpinionExample]
    :param train_curve_metrics: Optional bundle derived from the training split.
    :type train_curve_metrics: Optional[Dict[str, Any]]
    :returns: Curve payload containing evaluation (and optional training) metrics.
    :rtype: Optional[Dict[str, Any]]
    """
    eval_curve_metrics = _curve_payload(
        metrics_by_k,
        n_examples=len(eval_examples),
    )
    bundle: Optional[Dict[str, Any]] = None
    if eval_curve_metrics:
        bundle = {"eval": eval_curve_metrics}
        if train_curve_metrics:
            bundle["train"] = train_curve_metrics
    elif train_curve_metrics:
        bundle = {"train": train_curve_metrics}
    if bundle is None:
        return None
    bundle["metric"] = "mae_after"
    return bundle


def _select_best_k(
    metrics_by_k: Mapping[int, Mapping[str, float]],
    k_values: Sequence[int],
) -> int:
    """
    Return the best-performing ``k`` according to the R² metric.

    :param metrics_by_k: Mapping of ``k`` to evaluation metrics.
    :type metrics_by_k: Mapping[int, Mapping[str, float]]
    :param k_values: Ordered list of evaluated ``k`` values.
    :type k_values: Sequence[int]
    :returns: Selected ``k`` with maximal R², or the first requested ``k``.
    :rtype: int
    """
    if metrics_by_k:
        return max(metrics_by_k.items(), key=lambda item: item[1]["r2_after"])[0]
    return int(k_values[0])


def _evaluate_opinion_study(
    *,
    spec: OpinionSpec,
    context: OpinionEvaluationContext,
) -> None:
    """
    Run the full evaluation pipeline for a single opinion study.

    :param spec: Opinion study specification to evaluate.
    :type spec: OpinionSpec
    :param context: Shared evaluation context derived from CLI arguments.
    :type context: OpinionEvaluationContext
    """
    LOGGER.info("[OPINION] Study=%s (%s)", spec.key, spec.label)
    train_examples = collect_examples(
        context.dataset[TRAIN_SPLIT],
        spec=spec,
        extra_fields=context.extra_fields,
        max_examples=int(getattr(context.args, "knn_max_train", 0) or 0),
        seed=int(getattr(context.args, "knn_seed", 42)),
    )
    eval_examples = collect_examples(
        context.dataset[EVAL_SPLIT],
        spec=spec,
        extra_fields=context.extra_fields,
        max_examples=int(getattr(context.args, "eval_max", 0) or 0),
        seed=int(getattr(context.args, "knn_seed", 42)),
    )
    if not eval_examples:
        LOGGER.warning("[OPINION] No evaluation examples found for study=%s", spec.key)
        return

    train_examples = exclude_eval_participants(
        train_examples,
        eval_examples,
        logger=LOGGER,
        study_key=spec.key,
    )

    if not ensure_train_examples(
        train_examples,
        logger=LOGGER,
        message="[OPINION] No non-overlapping training examples remain for study=%s",
        args=(spec.key,),
    ):
        return

    index = build_index(
        examples=train_examples,
        feature_space=context.feature_space,
        spec=spec,
        seed=int(getattr(context.args, "knn_seed", 42)),
        metric=str(getattr(context.args, "knn_metric", "cosine")),
        word2vec_config=context.embedding_configs.word2vec,
        sentence_config=context.embedding_configs.sentence_transformer,
    )

    log_participant_counts(
        LOGGER,
        study_key=spec.key,
        train_count=len(train_examples),
        eval_count=len(eval_examples),
    )

    predictions = _evaluation_predictions(index, eval_examples, context.k_values)
    best_k = _select_best_k(predictions.metrics_by_k, context.k_values)
    output_payload = _OutputPayload(
        rows=predictions.rows,
        metrics_by_k=predictions.metrics_by_k,
        baseline=_baseline_metrics(eval_examples),
        best_k=best_k,
        curve_metrics=_curve_metrics_bundle(
            predictions.metrics_by_k,
            eval_examples,
            _training_curve_metrics(index, train_examples, context.k_values),
        ),
    )
    _write_outputs(
        context=_OutputContext(
            args=context.args,
            spec=spec,
            index=index,
            outputs_root=context.outputs_root,
        ),
        payload=output_payload,
    )


def run_opinion_eval(args) -> None:
    """
    Execute the post-study opinion index evaluation.

    :param args: Namespace object containing parsed command-line arguments.
    :type args: Any
    """
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("HF_HOME", args.cache_dir)

    dataset = load_dataset_source(args.dataset or DEFAULT_DATASET_SOURCE, args.cache_dir)
    specs = _resolve_requested_specs(args)
    extra_fields = _extra_fields_from_args(args)
    k_values = parse_k_values(args.knn_k, args.knn_k_sweep)
    LOGGER.info("[OPINION] Evaluating k values: %s", k_values)

    feature_space = str(getattr(args, "feature_space", "tfidf")).lower()
    word2vec_cfg = _resolve_word2vec_config(args, feature_space)
    sentence_cfg = _resolve_sentence_config(args, feature_space)

    outputs_root = Path(args.out_dir) / "opinion" / feature_space
    outputs_root.mkdir(parents=True, exist_ok=True)

    embedding_configs = OpinionEmbeddingConfigs(
        word2vec=word2vec_cfg,
        sentence_transformer=sentence_cfg,
    )

    evaluation_context = OpinionEvaluationContext(
        args=args,
        dataset=dataset,
        extra_fields=extra_fields,
        k_values=k_values,
        feature_space=feature_space,
        embedding_configs=embedding_configs,
        outputs_root=outputs_root,
    )

    for spec in specs:
        _evaluate_opinion_study(
            spec=spec,
            context=evaluation_context,
        )


__all__ = [
    *_OPINION_MODEL_EXPORTS,
    "OpinionSpec",
    "DEFAULT_SPECS",
    "run_opinion_eval",
]

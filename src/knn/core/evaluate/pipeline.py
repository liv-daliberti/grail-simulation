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

"""Evaluation entrypoints for the KNN baseline."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from common.evaluation.utils import compose_issue_slug, prepare_dataset, safe_div
from common.prompts.docs import merge_default_extra_fields

from ..data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    TRAIN_SPLIT,
    filter_split_for_participant_studies,
    issues_in_dataset,
    load_dataset_source,
)
from .curves import ValidationLogContext, curve_summary, log_validation_summary
from .dataset_eval import (
    DatasetEvaluationOptions,
    DatasetEvaluationRequest,
    evaluate_dataset_split,
    update_correct_counts,
)
from .filters import filter_split_for_issues
from .indexes import build_or_load_index
from .k_selection import parse_k_values, select_best_k
from .metrics import bootstrap_uncertainty
from .outputs import (
    IssueOutputArtifacts,
    IssueOutputMetadata,
    IssueOutputRequest,
    IssueOutputStats,
    write_issue_outputs,
)
from .provenance import collect_dataset_provenance, collect_repo_state
from .utils import split_tokens


@dataclass(frozen=True)
class IssueEvaluationMetadata:
    """Identifiers describing the current issue evaluation slice."""

    issue_slug: str
    dataset_source: str
    feature_space: str


@dataclass(frozen=True)
class IssueEvaluationDatasets:
    """Active dataset splits for evaluation."""

    train: Any
    eval: Any


@dataclass(frozen=True)
class IssueEvaluationSweep:
    """KNN sweep configuration shared across issuess."""

    k_values: Sequence[int]
    extra_fields: Sequence[str]


@dataclass(frozen=True)
class IssueFilters:
    """Filters applied when selecting dataset slices for evaluation."""

    available: Sequence[str]
    lookup: Mapping[str, str]
    train_issue_tokens: Sequence[str]
    train_study_tokens: Sequence[str]
    eval_study_tokens: Sequence[str]


@dataclass(frozen=True)
class PipelineResources:
    """Resources shared across issue evaluations."""

    dataset_source: str
    base_ds: Mapping[str, Any]
    base_provenance: Mapping[str, Any]
    repo_state: Mapping[str, Any]


@dataclass(frozen=True)
class IssueBuildInputs:
    """Context required to construct :class:`IssueEvaluationRequest` objects."""

    args: Any
    filters: IssueFilters
    resources: PipelineResources
    sweep: IssueEvaluationSweep


@dataclass(frozen=True)
class IssueEvaluationRequest:
    """Inputs required to evaluate and persist results for a single issue."""

    args: Any
    metadata: IssueEvaluationMetadata
    datasets: IssueEvaluationDatasets
    knn_index: Mapping[str, Any]
    sweep: IssueEvaluationSweep
    provenance: Mapping[str, Any]

    @property
    def issue_slug(self) -> str:
        """
        Return the slug identifying the issue slice being evaluated.

        :returns: Issue slug used to align outputs and logging.
        """
        return self.metadata.issue_slug

    @property
    def dataset_source(self) -> str:
        """
        Return the logical dataset source backing this evaluation.

        :returns: Source identifier describing the dataset loader in use.
        """
        return self.metadata.dataset_source

    @property
    def feature_space(self) -> str:
        """
        Return the feature-space label describing the representation in use.

        :returns: Normalised feature space evaluated for the current issue.
        """
        return self.metadata.feature_space

    @property
    def train_ds(self) -> Any:
        """
        Return the training split prepared for auxiliary diagnostics.

        :returns: Training dataset or ``None`` when not available.
        """
        return self.datasets.train

    @property
    def eval_ds(self) -> Any:
        """
        Return the evaluation split over which metrics are computed.

        :returns: Evaluation dataset yielding rows to score.
        """
        return self.datasets.eval

    @property
    def k_values(self) -> Sequence[int]:
        """
        Return the neighbourhood sizes included in the ``k`` sweep.

        :returns: Ordered list of ``k`` values evaluated for the issue.
        """
        return self.sweep.k_values

    @property
    def extra_fields(self) -> Sequence[str]:
        """
        Return the additional metadata fields to persist with evaluation outputs.

        :returns: Sequence of extra field names merged into each prediction row.
        """
        return self.sweep.extra_fields


@dataclass(frozen=True)
class PreparedIssueInputs:
    """Intermediate artefacts required to assemble an issue evaluation request."""

    issue_slug: str
    filtered_train: Any
    filtered_eval: Any
    current_train_issues: Sequence[str]
    provenance: Mapping[str, Any]


def _dataset_options(**overrides: Any) -> DatasetEvaluationOptions:
    """
    Return dataset evaluation options populated with ``overrides``.

    :param overrides: Keyword arguments overriding default evaluation options.
    :returns: Populated :class:`DatasetEvaluationOptions` instance.
    """

    return replace(DatasetEvaluationOptions(), **overrides)


def _positive_or_none(value: Optional[int]) -> Optional[int]:
    """
    Return ``value`` if strictly positive; otherwise ``None``.

    :param value: Candidate integer to validate.
    :returns: ``value`` when strictly positive, otherwise ``None``.
    """

    if value and value > 0:
        return int(value)
    return None


def _accuracy_for_k_values(
    per_k_stats: Mapping[int, Mapping[str, int]],
    k_values: Sequence[int],
) -> Dict[int, float]:
    """
    Compute accuracy by ``k`` from per-k statistics.

    :param per_k_stats: Aggregate counts keyed by ``k`` and metric label.
    :param k_values: Sequence of neighbourhood sizes to report.
    :returns: Mapping of ``k`` to accuracy ratios.
    """

    return {
        k: safe_div(per_k_stats[k]["correct"], per_k_stats[k]["eligible"])
        for k in k_values
    }


def _baseline_index_from_hist(gold_hist: Mapping[int, int]) -> Optional[int]:
    """
    Return the baseline index corresponding to the most frequent gold label.

    :param gold_hist: Histogram of gold-label counts keyed by index.
    :returns: Baseline index or ``None`` when the histogram is empty.
    """

    if not gold_hist:
        return None
    return max(gold_hist.items(), key=lambda kv: kv[1])[0]


def _issue_args(
    args: Any,
    available_issues: Sequence[str],
    issue_lookup: Mapping[str, str],
) -> tuple[Sequence[str], List[str]]:
    """
    Return training issue tokens and resolved evaluation issue list.

    :param args: CLI namespace supplying issue token arguments.
    :param available_issues: All issues exposed by the dataset.
    :param issue_lookup: Case-insensitive mapping from token to canonical issue.
    :returns: Tuple of train issue tokens and the resolved evaluation issues.
    """

    eval_tokens = split_tokens(getattr(args, "eval_issues", "")) or split_tokens(
        getattr(args, "issues", "")
    )
    train_tokens = split_tokens(getattr(args, "train_issues", ""))
    issues = _resolve_issue_list(eval_tokens, available_issues, issue_lookup)
    return train_tokens, issues


def _study_token_filters(args: Any) -> tuple[Sequence[str], Sequence[str]]:
    """
    Return train/eval participant study token filters extracted from ``args``.

    :param args: CLI namespace containing participant study flags.
    :returns: Tuple of train and eval participant study token sequences.
    """

    joint_tokens = split_tokens(getattr(args, "participant_studies", ""))
    train_tokens = split_tokens(getattr(args, "train_participant_studies", "")) or joint_tokens
    eval_tokens = split_tokens(getattr(args, "eval_participant_studies", "")) or joint_tokens
    return train_tokens, eval_tokens


def _evaluate_train_curve(
    request: IssueEvaluationRequest,
    k_values: Sequence[int],
    selection_method: str,
) -> Optional[Dict[str, Any]]:
    """
    Evaluate the training split (if available) for curve visualisation.

    :param request: Issue evaluation bundle describing the current issue.
    :param k_values: Sorted collection of neighbourhood sizes to score.
    :param selection_method: Strategy for selecting the best ``k``.
    :returns: Curve summary for the train split or ``None`` when unavailable.
    """

    if request.train_ds is None:
        return None

    train_summary = evaluate_dataset_split(
        DatasetEvaluationRequest(
            dataset=request.train_ds,
            k_values=k_values,
            knn_index=request.knn_index,
            extra_fields=request.extra_fields,
            metric=request.args.knn_metric,
            options=_dataset_options(
                capture_rows=False,
                log_label=f"train][{request.issue_slug}",
                max_examples=_positive_or_none(getattr(request.args, "train_curve_max", 0)),
                k_select_method=selection_method,
            ),
        )
    )
    train_accuracy_by_k = _accuracy_for_k_values(
        train_summary["per_k_stats"],
        k_values,
    )
    train_best_k = select_best_k(
        k_values,
        train_accuracy_by_k,
        method=selection_method,
    )
    return curve_summary(
        k_values=k_values,
        accuracy_by_k=train_accuracy_by_k,
        per_k_stats=train_summary["per_k_stats"],
        best_k=train_best_k,
        n_examples=train_summary["n_examples"],
    )


def evaluate_issue(request: IssueEvaluationRequest) -> None:
    """
    Evaluate a single issue slice and persist metrics, curves, and predictions.

    :param request: Evaluation bundle describing the issue to score.
    :returns: ``None``. Results are persisted via :func:`write_issue_outputs`.
    """

    k_values = sorted({int(k) for k in request.k_values if int(k) > 0})
    selection_method = getattr(request.args, "k_select_method", "max")
    eval_summary = evaluate_dataset_split(
        DatasetEvaluationRequest(
            dataset=request.eval_ds,
            k_values=k_values,
            knn_index=request.knn_index,
            extra_fields=request.extra_fields,
            metric=request.args.knn_metric,
            options=_dataset_options(
                capture_rows=True,
                log_label=f"eval][{request.issue_slug}",
                max_examples=_positive_or_none(getattr(request.args, "eval_max", 0)),
                k_select_method=selection_method,
            ),
        )
    )

    accuracy_by_k = _accuracy_for_k_values(
        eval_summary["per_k_stats"],
        k_values,
    )
    best_k = select_best_k(k_values, accuracy_by_k, method=selection_method)
    update_correct_counts(
        eval_summary["rows"],
        best_k,
        eval_summary["bucket_stats"],
        eval_summary["single_multi_stats"],
    )
    eval_curve = curve_summary(
        k_values=k_values,
        accuracy_by_k=accuracy_by_k,
        per_k_stats=eval_summary["per_k_stats"],
        best_k=best_k,
        n_examples=eval_summary["n_examples"],
    )

    curve_metrics: Dict[str, Any] = {"eval": eval_curve}
    train_curve = _evaluate_train_curve(request, k_values, selection_method)
    if train_curve:
        curve_metrics["train"] = train_curve

    baseline_index = _baseline_index_from_hist(eval_summary["gold_hist"])
    uncertainty = bootstrap_uncertainty(
        rows=eval_summary["rows"],
        best_k=best_k,
        baseline_index=baseline_index,
        replicates=int(getattr(request.args, "bootstrap_replicates", 500) or 0),
        seed=int(getattr(request.args, "bootstrap_seed", 2024) or 2024),
    )

    write_issue_outputs(
        IssueOutputRequest(
            args=request.args,
            metadata=IssueOutputMetadata(
                issue_slug=request.issue_slug,
                feature_space=request.feature_space,
                dataset_source=request.dataset_source,
            ),
            stats=IssueOutputStats(
                k_values=k_values,
                accuracy_by_k=accuracy_by_k,
                best_k=best_k,
                bucket_stats=eval_summary["bucket_stats"],
                single_multi_stats=eval_summary["single_multi_stats"],
                gold_hist=eval_summary["gold_hist"],
                per_k_stats=eval_summary["per_k_stats"],
            ),
            artifacts=IssueOutputArtifacts(
                rows=eval_summary["rows"],
                extra_fields=request.extra_fields,
                curve_metrics=curve_metrics,
                provenance=request.provenance,
                uncertainty=uncertainty,
            ),
        )
    )
    log_validation_summary(
        ValidationLogContext(
            issue_slug=request.issue_slug,
            feature_space=request.feature_space,
            best_k=best_k,
            accuracy_by_k=accuracy_by_k,
            per_k_stats=eval_summary["per_k_stats"],
            n_examples=eval_summary["n_examples"],
        )
    )


def _resolve_issue_list(
    tokens: Sequence[str],
    fallback: Sequence[str],
    lookup: Mapping[str, str],
) -> List[str]:
    """
    Normalise user-provided issue tokens against ``fallback``.

    :param tokens: Raw issue tokens provided via CLI.
    :param fallback: Default issue list when no tokens are supplied.
    :param lookup: Mapping from lowercase token to canonical issue name.
    :returns: Resolved issue list, falling back when tokens are unknown.
    """

    if not tokens:
        return list(fallback)
    if any(token.lower() == "all" for token in tokens):
        return list(fallback)
    resolved: List[str] = []
    for token in tokens:
        match = lookup.get(token.lower())
        if match:
            resolved.append(match)
        else:
            logging.warning("[KNN] Unknown issue token '%s'; skipping.", token)
    return resolved or list(fallback)


def _prepare_filtered_splits(
    issue: str,
    current_train_issues: Sequence[str],
    base_ds,
    filters: IssueFilters,
):
    """
    Return filtered train/eval splits based on issue and participant filters.

    :param issue: Canonical evaluation issue identifier.
    :param current_train_issues: Issues used when filtering the train split.
    :param base_ds: Original dataset splits keyed by split name.
    :param filters: Active issue and participant study filters.
    :returns: Tuple of filtered train and eval splits.
    """

    train_split = base_ds.get(TRAIN_SPLIT)
    eval_split = base_ds.get(EVAL_SPLIT)
    filtered_train = train_split
    if current_train_issues and train_split is not None:
        filtered_train = filter_split_for_issues(train_split, current_train_issues)
    filtered_eval = eval_split
    if issue != "all" and eval_split is not None:
        filtered_eval = filter_split_for_issues(eval_split, [issue])

    filtered_train = filter_split_for_participant_studies(
        filtered_train,
        filters.train_study_tokens,
    )
    filtered_eval = filter_split_for_participant_studies(
        filtered_eval,
        filters.eval_study_tokens,
    )
    return filtered_train, filtered_eval


def _prepare_issue_request_inputs(
    issue: str,
    context: IssueBuildInputs,
) -> Optional[PreparedIssueInputs]:
    """Return datasets and metadata required to evaluate ``issue``."""

    filters = context.filters
    if filters.train_issue_tokens:
        current_train_issues = _resolve_issue_list(
            filters.train_issue_tokens,
            filters.available,
            filters.lookup,
        )
    else:
        current_train_issues = [issue] if issue != "all" else []

    filtered_train, filtered_eval = _prepare_filtered_splits(
        issue,
        current_train_issues,
        context.resources.base_ds,
        filters,
    )

    if filtered_eval is None or len(filtered_eval) == 0:
        logging.warning(
            "[KNN] Skipping issue=%s (no evaluation rows after filters: "
            "eval_studies=%s eval_issue=%s)",
            issue,
            ",".join(filters.eval_study_tokens) or "all",
            issue,
        )
        return None
    if context.args.fit_index and (filtered_train is None or len(filtered_train) == 0):
        logging.warning(
            "[KNN] Skipping issue=%s (no training rows after filters and "
            "--fit-index enabled)",
            issue,
        )
        return None

    issue_slug = compose_issue_slug(issue, filters.eval_study_tokens)
    provenance = {
        "dataset": {
            "source": context.resources.dataset_source,
            "base": context.resources.base_provenance,
            "active": collect_dataset_provenance(
                {
                    TRAIN_SPLIT: filtered_train,
                    EVAL_SPLIT: filtered_eval,
                }
            ),
            "filters": {
                "train_issues": current_train_issues,
                "eval_issue": issue,
                "train_participant_studies": list(filters.train_study_tokens),
                "eval_participant_studies": list(filters.eval_study_tokens),
            },
        },
        "code": context.resources.repo_state,
    }

    return PreparedIssueInputs(
        issue_slug=issue_slug,
        filtered_train=filtered_train,
        filtered_eval=filtered_eval,
        current_train_issues=current_train_issues,
        provenance=provenance,
    )


def _build_issue_request(
    issue: str,
    context: IssueBuildInputs,
) -> Optional[IssueEvaluationRequest]:
    """
    Prepare an :class:`IssueEvaluationRequest` for ``issue`` or ``None``.

    :param issue: Target issue token (or ``all``) requested on the CLI.
    :param context: Shared build inputs collected for each issue.
    :returns: Populated request or ``None`` when no evaluation rows are available.
    """

    prepared = _prepare_issue_request_inputs(issue, context)
    if prepared is None:
        return None

    knn_index = build_or_load_index(
        train_ds=prepared.filtered_train,
        issue_slug=prepared.issue_slug,
        extra_fields=context.sweep.extra_fields,
        args=context.args,
    )

    logging.info(
        "[KNN] issue=%s train_rows=%d eval_rows=%d train_studies=%s "
        "eval_studies=%s train_issues=%s",
        prepared.issue_slug,
        len(prepared.filtered_train) if prepared.filtered_train is not None else 0,
        len(prepared.filtered_eval),
        ",".join(context.filters.train_study_tokens) or "all",
        ",".join(context.filters.eval_study_tokens) or "all",
        ",".join(prepared.current_train_issues)
        or ("(inherit:" + issue + ")"),
    )

    return IssueEvaluationRequest(
        args=context.args,
        metadata=IssueEvaluationMetadata(
            issue_slug=prepared.issue_slug,
            dataset_source=context.resources.dataset_source,
            feature_space=str(knn_index.get("feature_space", "tfidf")).lower(),
        ),
        datasets=IssueEvaluationDatasets(
            train=(
                prepared.filtered_train
                if prepared.filtered_train is not None
                and len(prepared.filtered_train)
                else None
            ),
            eval=prepared.filtered_eval,
        ),
        knn_index=knn_index,
        sweep=context.sweep,
        provenance=prepared.provenance,
    )


def run_eval(args) -> None:
    """
    Evaluate the KNN baseline across the issues specified on the CLI.

    :param args: Namespace returned by :func:`knn.cli.build_parser`, including dataset,
        feature-space, and sweep configuration.
    :returns: ``None``. Evaluation artefacts are persisted via downstream helpers.
    """

    dataset_source, base_ds, available_issues = prepare_dataset(
        dataset=getattr(args, "dataset", None),
        default_source=DEFAULT_DATASET_SOURCE,
        cache_dir=args.cache_dir,
        loader=load_dataset_source,
        issue_lookup=issues_in_dataset,
    )
    issue_lookup = {issue.lower(): issue for issue in available_issues}

    train_issue_tokens, issues = _issue_args(args, available_issues, issue_lookup)
    train_study_tokens, eval_study_tokens = _study_token_filters(args)

    k_values = parse_k_values(args.knn_k, args.knn_k_sweep)
    logging.info("[KNN] Evaluating k values: %s", k_values)

    context = IssueBuildInputs(
        args=args,
        filters=IssueFilters(
            available=available_issues,
            lookup=issue_lookup,
            train_issue_tokens=train_issue_tokens,
            train_study_tokens=train_study_tokens,
            eval_study_tokens=eval_study_tokens,
        ),
        resources=PipelineResources(
            dataset_source=dataset_source,
            base_ds=base_ds,
            base_provenance=collect_dataset_provenance(base_ds),
            repo_state=collect_repo_state(),
        ),
        sweep=IssueEvaluationSweep(
            k_values=k_values,
            extra_fields=merge_default_extra_fields(split_tokens(args.knn_text_fields)),
        ),
    )

    if base_ds.get(TRAIN_SPLIT) is None or base_ds.get(EVAL_SPLIT) is None:
        raise RuntimeError(
            f"Dataset '{dataset_source}' must expose '{TRAIN_SPLIT}' and '{EVAL_SPLIT}' splits."
        )

    for issue in issues:
        request = _build_issue_request(issue, context)
        if request is None:
            continue
        evaluate_issue(request)


__all__ = ["evaluate_issue", "run_eval"]

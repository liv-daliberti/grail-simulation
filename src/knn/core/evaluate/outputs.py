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

"""Persistence helpers for KNN evaluation artefacts."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from common.evaluation.utils import safe_div

from ..data import EVAL_SPLIT
from .curves import plot_elbow
from .utils import BUCKET_LABELS


@dataclass(frozen=True)
class IssueOutputMetadata:
    """Identifiers describing the evaluation slice being written."""

    issue_slug: str
    feature_space: str
    dataset_source: str


@dataclass(frozen=True)
class IssueOutputStats:
    """Aggregated statistics collected during evaluation."""

    k_values: Sequence[int]
    accuracy_by_k: Mapping[int, float]
    best_k: int
    bucket_stats: Mapping[str, Mapping[str, int]]
    single_multi_stats: Mapping[str, int]
    gold_hist: Mapping[int, int]
    per_k_stats: Mapping[int, Mapping[str, int]]


@dataclass(frozen=True)
class IssueOutputArtifacts:
    """Raw artefacts that need to be persisted alongside metrics."""

    rows: Sequence[Dict[str, Any]]
    extra_fields: Sequence[str]
    curve_metrics: Mapping[str, Any]
    provenance: Mapping[str, Any]
    uncertainty: Optional[Mapping[str, Any]] = None


@dataclass(frozen=True)
class IssueOutputRequest:
    """Bundle capturing everything needed to persist issue-level outputs."""

    args: Any
    metadata: IssueOutputMetadata
    stats: IssueOutputStats
    artifacts: IssueOutputArtifacts

    @property
    def issue_slug(self) -> str:
        """
        Return the slug that uniquely identifies the evaluated issue slice.

        :returns: Issue identifier used when naming artefacts and directories.
        :rtype: str
        """
        return self.metadata.issue_slug

    @property
    def feature_space(self) -> str:
        """
        Return the feature-space label associated with the evaluation request.

        :returns: Feature space (e.g. embedding family) used for the KNN model.
        :rtype: str
        """
        return self.metadata.feature_space

    @property
    def dataset_source(self) -> str:
        """
        Return the dataset source describing where evaluation rows originated.

        :returns: Logical dataset source name (typically matches the loader key).
        :rtype: str
        """
        return self.metadata.dataset_source

    @property
    def rows(self) -> Sequence[Dict[str, Any]]:
        """
        Return the per-example evaluation rows captured during scoring.

        :returns: Iterable of rows containing predictions, gold indices, and metadata.
        :rtype: Sequence[Dict[str, Any]]
        """
        return self.artifacts.rows

    @property
    def k_values(self) -> Sequence[int]:
        """
        Return the ordered list of ``k`` values considered during evaluation.

        :returns: Integers representing the neighbourhood sizes that were swept.
        :rtype: Sequence[int]
        """
        return self.stats.k_values

    @property
    def accuracy_by_k(self) -> Mapping[int, float]:
        """
        Return the accuracy achieved for each evaluated ``k`` value.

        :returns: Mapping from ``k`` to accuracy computed over eligible examples.
        :rtype: Mapping[int, float]
        """
        return self.stats.accuracy_by_k

    @property
    def best_k(self) -> int:
        """
        Return the selected ``k`` that maximises the evaluation objective.

        :returns: The neighbourhood size chosen as the best performer.
        :rtype: int
        """
        return self.stats.best_k

    @property
    def bucket_stats(self) -> Mapping[str, Mapping[str, int]]:
        """
        Return aggregated bucket-level counts collected during evaluation.

        :returns: Nested mapping containing per-bucket totals, eligibility, and correctness.
        :rtype: Mapping[str, Mapping[str, int]]
        """
        return self.stats.bucket_stats

    @property
    def single_multi_stats(self) -> Mapping[str, int]:
        """
        Return statistics that disaggregate performance by slate cardinality.

        :returns: Mapping with counts and correctness for single-option and multi-option rows.
        :rtype: Mapping[str, int]
        """
        return self.stats.single_multi_stats

    @property
    def gold_hist(self) -> Mapping[int, int]:
        """
        Return histogram data describing how frequently each gold index appears.

        :returns: Mapping from gold index to occurrence count across eligible rows.
        :rtype: Mapping[int, int]
        """
        return self.stats.gold_hist

    @property
    def per_k_stats(self) -> Mapping[int, Mapping[str, int]]:
        """
        Return detailed statistics collected for each evaluated ``k`` value.

        :returns: Mapping from ``k`` to counters such as ``eligible`` and ``correct``.
        :rtype: Mapping[int, Mapping[str, int]]
        """
        return self.stats.per_k_stats

    @property
    def extra_fields(self) -> Sequence[str]:
        """
        Return the additional metadata fields persisted with per-example outputs.

        :returns: Sequence of field names carried through from the dataset rows.
        :rtype: Sequence[str]
        """
        return self.artifacts.extra_fields

    @property
    def curve_metrics(self) -> Mapping[str, Any]:
        """
        Return the curve summaries captured for train and evaluation splits.

        :returns: Mapping from split name to elbow/accuracy curve metadata.
        :rtype: Mapping[str, Any]
        """
        return self.artifacts.curve_metrics

    @property
    def provenance(self) -> Mapping[str, Any]:
        """
        Return provenance information describing how the artefacts were produced.

        :returns: Metadata bundle containing dataset lineage, hyperparameters, and git state.
        :rtype: Mapping[str, Any]
        """
        return self.artifacts.provenance

    @property
    def uncertainty(self) -> Optional[Mapping[str, Any]]:
        """
        Return bootstrap uncertainty metrics, if they were computed.

        :returns: Mapping with uncertainty summaries or ``None`` when not requested.
        :rtype: Optional[Mapping[str, Any]]
        """
        return self.artifacts.uncertainty


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Performance metrics derived from the best ``k`` evaluation."""

    eligible_overall: int
    accuracy_all_rows: float
    best_accuracy: float


@dataclass(frozen=True)
class BucketSummary:
    """Serialised statistics over the options buckets."""

    position_totals: Mapping[str, int]
    accuracy_by_options: Mapping[str, float]


@dataclass(frozen=True)
class BaselineSummary:
    """Baseline comparisons for KNN evaluation."""

    index: Optional[int]
    accuracy: float
    gold_distribution: Mapping[str, int]
    random_accuracy: float


@dataclass(frozen=True)
class OutputPaths:
    """File paths persisted alongside the metrics document."""

    elbow: Path
    curve_json: Path


def resolve_reports_dir(out_dir: Path) -> Path:
    """
    Resolve the canonical reports directory corresponding to ``out_dir``.

    :param out_dir: Directory containing pipeline artefacts (typically under ``models/knn``).
    :returns: Path pointing at the repository-level ``reports`` directory.
    """

    resolved = out_dir.resolve()
    ancestors = [resolved, *resolved.parents]
    root_dir: Optional[Path] = None

    for ancestor in ancestors:
        if ancestor.name == "models":
            root_dir = ancestor.parent
            break
    if root_dir is None:
        for ancestor in ancestors:
            if ancestor.name in {"knn", "xgb"}:
                root_dir = ancestor.parent
                break
    if root_dir is None and resolved.parents:
        root_dir = resolved.parents[-1]
    if root_dir is None:
        root_dir = resolved
    return root_dir / "reports"


def _write_rows_jsonl(request: IssueOutputRequest, issue_dir: Path) -> Path:
    """Write per-example predictions to JSONL and return the output path."""

    output_path = issue_dir / f"knn_eval_{request.issue_slug}_{EVAL_SPLIT}.jsonl"
    with open(output_path, "w", encoding="utf-8") as handle:
        for row in request.rows:
            predictions = {
                str(k): (int(v) if v is not None else None)
                for k, v in row["predictions_by_k"].items()
            }
            best_pred = row["predictions_by_k"].get(request.best_k)
            record = {
                "knn_pred_index": int(best_pred) if best_pred is not None else None,
                "gold_index": row["gold_index"],
                "n_options": row["n_options"],
                "correct": bool(best_pred is not None and int(best_pred) == row["gold_index"]),
                "eligible": row["eligible"],
                "position_index": row["position_index"],
                "position_bucket": row["position_bucket"],
                "issue": request.issue_slug
                if request.issue_slug != "all"
                else row.get("issue_value"),
                "predictions_by_k": predictions,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output_path


def _serialise_bucket_stats(
    bucket_stats: Mapping[str, Mapping[str, int]],
) -> tuple[Dict[str, int], Dict[str, float]]:
    """Return serialisable bucket statistics for inclusion in metrics."""

    position_totals = {
        bucket: int(bucket_stats["position_seen"][bucket])
        for bucket in BUCKET_LABELS
    }
    accuracy_by_options = {
        bucket: safe_div(
            bucket_stats["options_correct"][bucket],
            bucket_stats["options_eligible"][bucket],
        )
        for bucket in BUCKET_LABELS
    }
    return position_totals, accuracy_by_options


def _baseline_summary(
    gold_hist: Mapping[int, int],
    eligible_overall: int,
) -> tuple[Optional[int], float, Dict[str, int]]:
    """Return baseline statistics describing the most frequent gold index."""

    distribution = {str(k): int(v) for k, v in sorted(gold_hist.items())}
    if not gold_hist or eligible_overall <= 0:
        return None, 0.0, distribution
    top_idx, top_count = max(gold_hist.items(), key=lambda kv: kv[1])
    baseline_accuracy = safe_div(top_count, eligible_overall)
    return top_idx, baseline_accuracy, distribution


def _random_baseline_accuracy(single_multi_stats: Mapping[str, int]) -> float:
    """Return the random-choice baseline accuracy derived from slate sizes."""

    random_sum = float(single_multi_stats.get("rand_inverse_sum", 0.0))
    random_count = int(single_multi_stats.get("rand_inverse_count", 0))
    return safe_div(random_sum, random_count)


def _write_per_k_outputs(
    request: IssueOutputRequest,
    *,
    issue_dir: Path,
    elbow_path: Path,
) -> None:
    """Persist per-k prediction JSONL and metric summaries."""

    for k in request.k_values:
        k_int = int(k)
        k_dir = issue_dir / f"k-{k_int}"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_predictions = k_dir / f"predictions_{request.issue_slug}_{EVAL_SPLIT}.jsonl"
        with open(k_predictions, "w", encoding="utf-8") as handle:
            for row in request.rows:
                pred_value = row["predictions_by_k"].get(k_int)
                record = {
                    "k": k_int,
                    "knn_pred_index": int(pred_value) if pred_value is not None else None,
                    "gold_index": row["gold_index"],
                    "eligible": row["eligible"],
                    "correct": bool(
                        pred_value is not None and int(pred_value) == row["gold_index"]
                    ),
                    "n_options": row["n_options"],
                    "position_index": row["position_index"],
                    "issue": request.issue_slug
                    if request.issue_slug != "all"
                    else row.get("issue_value"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        k_stats = request.per_k_stats[k_int]
        k_metrics: Dict[str, Any] = {
            "model": "knn",
            "feature_space": request.feature_space,
            "dataset": request.dataset_source,
            "issue": request.issue_slug,
            "split": EVAL_SPLIT,
            "k": k_int,
            "n_total": int(len(request.rows)),
            "n_eligible": int(k_stats["eligible"]),
            "n_correct": int(k_stats["correct"]),
            "accuracy": float(request.accuracy_by_k.get(k_int, 0.0)),
            "elbow_plot": str(elbow_path),
            "provenance": request.provenance,
        }
        if request.uncertainty:
            k_metrics["uncertainty"] = request.uncertainty
        metrics_path = k_dir / f"metrics_{request.issue_slug}_{EVAL_SPLIT}.json"
        with open(metrics_path, "w", encoding="utf-8") as handle:
            json.dump(k_metrics, handle, ensure_ascii=False, indent=2)


def _compute_performance(request: IssueOutputRequest) -> PerformanceSnapshot:
    """Return aggregate accuracy figures for the best ``k``."""

    best_accuracy = request.accuracy_by_k.get(request.best_k, 0.0)
    best_k_stats = request.per_k_stats.get(int(request.best_k), {})
    eligible_overall = int(best_k_stats.get("eligible", 0))
    correct_eligible_best = int(best_k_stats.get("correct", 0))
    accuracy_all_rows = safe_div(correct_eligible_best, len(request.rows))
    return PerformanceSnapshot(
        eligible_overall=eligible_overall,
        accuracy_all_rows=accuracy_all_rows,
        best_accuracy=best_accuracy,
    )


def _bucket_summary_from_request(request: IssueOutputRequest) -> BucketSummary:
    """Return serialised bucket statistics from the evaluation summary."""

    position_stats, accuracy_by_options = _serialise_bucket_stats(request.bucket_stats)
    return BucketSummary(
        position_totals=position_stats,
        accuracy_by_options=accuracy_by_options,
    )


def _build_baseline_summary(
    request: IssueOutputRequest,
    eligible_overall: int,
) -> BaselineSummary:
    """Return baseline metrics derived from evaluation artefacts."""

    baseline_index, baseline_accuracy, gold_distribution = _baseline_summary(
        request.gold_hist,
        eligible_overall,
    )
    random_baseline = _random_baseline_accuracy(request.single_multi_stats)
    return BaselineSummary(
        index=baseline_index,
        accuracy=baseline_accuracy,
        gold_distribution=gold_distribution,
        random_accuracy=random_baseline,
    )


def _prepare_output_paths(request: IssueOutputRequest, issue_dir: Path) -> OutputPaths:
    """Create output directories, elbow plot, and curve JSON artefact."""

    reports_root = resolve_reports_dir(Path(request.args.out_dir))
    reports_dir = reports_root / "knn" / request.feature_space
    reports_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = reports_dir / f"elbow_{request.issue_slug}.png"
    plot_elbow(
        request.k_values,
        request.accuracy_by_k,
        request.best_k,
        elbow_path,
        data_split=EVAL_SPLIT,
    )

    curve_json = issue_dir / f"knn_curves_{request.issue_slug}.json"
    with open(curve_json, "w", encoding="utf-8") as handle:
        json.dump(request.curve_metrics, handle, ensure_ascii=False, indent=2)
    return OutputPaths(elbow=elbow_path, curve_json=curve_json)


def _build_metrics_payload(
    request: IssueOutputRequest,
    performance: PerformanceSnapshot,
    bucket_summary: BucketSummary,
    baselines: BaselineSummary,
    paths: OutputPaths,
) -> Dict[str, Any]:
    """Assemble the metrics document written alongside evaluation outputs."""

    payload: Dict[str, Any] = {
        "model": "knn",
        "feature_space": request.feature_space,
        "dataset": request.dataset_source,
        "issue": request.issue_slug,
        "split": EVAL_SPLIT,
        "n_total": int(len(request.rows)),
        "n_eligible": int(performance.eligible_overall),
        "accuracy_overall": performance.best_accuracy,
        "accuracy_overall_all_rows": float(performance.accuracy_all_rows),
        "accuracy_by_k": {
            str(k): float(request.accuracy_by_k[k]) for k in request.k_values
        },
        "best_k": int(request.best_k),
        "k_select_method": str(getattr(request.args, "k_select_method", "max")),
        "position_stats": dict(bucket_summary.position_totals),
        "by_n_options": {
            bucket: {
                "hist_seen": int(request.bucket_stats["options_seen"][bucket]),
                "hist_eligible": int(request.bucket_stats["options_eligible"][bucket]),
                "hist_correct": int(request.bucket_stats["options_correct"][bucket]),
                "accuracy": bucket_summary.accuracy_by_options[bucket],
            }
            for bucket in BUCKET_LABELS
        },
        "split_single_vs_multi": {
            "n_single": int(request.single_multi_stats["seen_single"]),
            "n_multi": int(request.single_multi_stats["seen_multi"]),
            "eligible_single": int(request.single_multi_stats["elig_single"]),
            "eligible_multi": int(request.single_multi_stats["elig_multi"]),
            "accuracy_single": safe_div(
                request.single_multi_stats["corr_single"],
                request.single_multi_stats["elig_single"],
            ),
            "accuracy_multi": safe_div(
                request.single_multi_stats["corr_multi"],
                request.single_multi_stats["elig_multi"],
            ),
        },
        "gold_index_distribution": dict(baselines.gold_distribution),
        "baseline_most_frequent_gold_index": {
            "top_index": baselines.index,
            "count": int(request.gold_hist.get(baselines.index, 0))
            if baselines.index is not None
            else 0,
            "accuracy": baselines.accuracy,
        },
        "random_baseline_expected_accuracy": baselines.random_accuracy,
        "knn_hparams": {
            "k": int(request.args.knn_k),
            "k_sweep": [int(k) for k in request.k_values],
            "metric": request.args.knn_metric,
            "fit_index": bool(request.args.fit_index),
            "save_index": request.args.save_index or "",
            "load_index": request.args.load_index or "",
            "text_fields": list(request.extra_fields),
        },
        "elbow_plot": str(paths.elbow),
        "per_k_directories": {
            str(k): str((Path(request.args.out_dir) / request.issue_slug / f"k-{int(k)}"))
            for k in request.k_values
        },
        "curve_metrics": request.curve_metrics,
        "curve_metrics_path": str(paths.curve_json),
        "notes": "Accuracy computed over eligible rows (gold_index>0).",
        "provenance": request.provenance,
    }
    if request.uncertainty:
        model_uncertainty = request.uncertainty.get("model")
        baseline_uncertainty = request.uncertainty.get("baseline")
        if model_uncertainty:
            payload["accuracy_ci_95"] = model_uncertainty.get("ci95")
            payload["accuracy_uncertainty"] = model_uncertainty
        if baseline_uncertainty:
            payload["baseline_ci_95"] = baseline_uncertainty.get("ci95")
            payload["baseline_uncertainty"] = baseline_uncertainty
        payload["uncertainty"] = request.uncertainty
    return payload


def write_issue_outputs(request: IssueOutputRequest) -> None:
    """
    Persist evaluation artefacts and per-``k`` directories for an issue.

    :param request: Issue output bundle describing what should be written.
    """

    issue_dir = Path(request.args.out_dir) / request.issue_slug
    issue_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = _write_rows_jsonl(request, issue_dir)

    paths = _prepare_output_paths(request, issue_dir)
    performance = _compute_performance(request)
    bucket_summary = _bucket_summary_from_request(request)
    baselines = _build_baseline_summary(request, performance.eligible_overall)
    metrics = _build_metrics_payload(
        request,
        performance,
        bucket_summary,
        baselines,
        paths,
    )
    metrics_path = issue_dir / f"knn_eval_{request.issue_slug}_{EVAL_SPLIT}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    _write_per_k_outputs(
        request,
        issue_dir=issue_dir,
        elbow_path=paths.elbow,
    )

    logging.info(
        "[DONE][%s] split=%s n=%d eligible=%d knn_acc=%.4f (best_k=%d)",
        request.issue_slug,
        EVAL_SPLIT,
        len(request.rows),
        performance.eligible_overall,
        performance.best_accuracy,
        request.best_k,
    )
    logging.info("[WROTE] per-example: %s", out_jsonl)
    logging.info("[WROTE] metrics: %s", metrics_path)
    logging.info("[WROTE] curves: %s", paths.curve_json)


__all__ = ["IssueOutputRequest", "resolve_reports_dir", "write_issue_outputs"]

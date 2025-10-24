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

"""Evaluation loop and metrics for the KNN baseline."""

from __future__ import annotations

# pylint: disable=line-too-long,too-many-arguments,too-many-branches,too-many-lines,too-many-locals,too-many-statements

import json
import logging
import re
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from numpy.random import default_rng

from common.embeddings import SentenceTransformerConfig
from common.eval_utils import compose_issue_slug, prepare_dataset, safe_div

try:  # pragma: no cover - optional dependency
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from .data import (
    DEFAULT_DATASET_SOURCE,
    EVAL_SPLIT,
    SOLUTION_COLUMN,
    TRAIN_SPLIT,
    filter_split_for_participant_studies,
    issues_in_dataset,
    load_dataset_source,
)
from .features import Word2VecConfig, extract_slate_items
from .index import (
    SlateQueryConfig,
    build_tfidf_index,
    build_sentence_transformer_index,
    build_word2vec_index,
    knn_predict_among_slate_multi,
    load_tfidf_index,
    load_sentence_transformer_index,
    load_word2vec_index,
    save_tfidf_index,
    save_sentence_transformer_index,
    save_word2vec_index,
)

BUCKET_LABELS = ["unknown", "1", "2", "3", "4", "5+"]

def _split_tokens(raw: Optional[str]) -> List[str]:
    """
    Return a list of comma-separated tokens with whitespace trimmed.

    :param raw: Raw comma-separated string or ``None`` when no tokens are supplied.

    :type raw: Optional[str]

    :returns: a list of comma-separated tokens with whitespace trimmed

    :rtype: List[str]

    """
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]

@lru_cache(maxsize=1)
def _collect_repo_state() -> Dict[str, Any]:
    """
    Return git provenance for the current repository.

    :returns: git provenance for the current repository

    :rtype: Dict[str, Any]

    """
    repo_root = Path(__file__).resolve().parents[2]

    def _run_git(args: Sequence[str]) -> Optional[str]:
        """
        Execute ``git`` with ``args`` and capture standard output.

        :param args: Sequence of command-line arguments passed to ``git``.
        :type args: Sequence[str]
        :returns: Stripped stdout when the command succeeds, otherwise ``None``.
        :rtype: Optional[str]
        """
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, ValueError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--short"])
    dirty = bool(status)
    return {
        "git_commit": commit or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": dirty,
        "git_status": status or "",
    }

def _filter_split_for_issues(split_ds, issues: Sequence[str]):
    """
    Return ``split_ds`` filtered to the requested issue tokens.

    :param split_ds: Dataset split object being filtered or transformed.

    :type split_ds: Any

    :param issues: Iterable of issue identifiers used to filter the dataset.

    :type issues: Sequence[str]

    :returns: ``split_ds`` filtered to the requested issue tokens

    :rtype: Any

    """
    normalized = {token.strip().lower() for token in issues if token.strip()}
    if not normalized:
        return split_ds
    if "issue" not in split_ds.column_names:
        return split_ds

    def _match_issue(row: Mapping[str, Any]) -> bool:
        """
        Determine whether a row's ``issue`` matches the normalised filter set.

        :param row: Dataset example retrieved from ``split_ds``.
        :type row: Mapping[str, Any]
        :returns: ``True`` when the row belongs to the requested issue slice.
        :rtype: bool
        """
        value = row.get("issue")
        return str(value).strip().lower() in normalized

    return split_ds.filter(_match_issue)

def _dataset_split_provenance(split_ds) -> Dict[str, Any]:
    """
    Return provenance metadata for a single HF dataset split.

    :param split_ds: Dataset split object being filtered or transformed.

    :type split_ds: Any

    :returns: provenance metadata for a single HF dataset split

    :rtype: Dict[str, Any]

    """
    provenance: Dict[str, Any] = {
        "num_rows": int(len(split_ds)),
        "fingerprint": getattr(split_ds, "_fingerprint", None),
    }
    info = getattr(split_ds, "info", None)
    revision = getattr(info, "dataset_revision", None) if info is not None else None
    if revision:
        provenance["dataset_revision"] = revision
    return provenance

def _collect_dataset_provenance(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return provenance metadata for all splits contained in ``dataset``.

    :param dataset: Dataset object containing the splits required for evaluation.

    :type dataset: Mapping[str, Any]

    :returns: provenance metadata for all splits contained in ``dataset``

    :rtype: Dict[str, Any]

    """
    splits: Dict[str, Any] = {}
    revision: Optional[str] = None
    for split_name, split_ds in dataset.items():
        split_info = _dataset_split_provenance(split_ds)
        splits[split_name] = split_info
        if revision is None and split_info.get("dataset_revision"):
            revision = split_info["dataset_revision"]
    return {
        "dataset_revision": revision,
        "splits": splits,
    }

def _group_key_for_example(example: Mapping[str, Any], fallback_index: int) -> str:
    """
    Return a stable grouping key used for bootstrap resampling.

    :param example: Single dataset example under inspection.

    :type example: Mapping[str, Any]

    :param fallback_index: Secondary index consulted when the primary lookup fails.

    :type fallback_index: int

    :returns: a stable grouping key used for bootstrap resampling

    :rtype: str

    """
    urlid = str(example.get("urlid") or "").strip()
    if urlid and urlid.lower() != "nan":
        return f"urlid::{urlid}"
    participant = str(example.get("participant_id") or "").strip()
    if participant and participant.lower() != "nan":
        return f"participant::{participant}"
    session = str(example.get("session_id") or "").strip()
    if session and session.lower() != "nan":
        return f"session::{session}"
    return f"row::{fallback_index}"

def _accuracy_for_rows(rows: Sequence[Mapping[str, Any]], k_val: int) -> float:
    """
    Return accuracy for ``rows`` using predictions at ``k_val``.

    :param rows: Iterable of evaluation rows or metrics to analyse.

    :type rows: Sequence[Mapping[str, Any]]

    :param k_val: Specific ``k`` value under evaluation.

    :type k_val: int

    :returns: accuracy for ``rows`` using predictions at ``k_val``

    :rtype: float

    """
    if not rows:
        return 0.0
    correct = 0
    total = 0
    for row in rows:
        if not row.get("eligible"):
            continue
        total += 1
        pred = row["predictions_by_k"].get(k_val)
        if pred is not None and int(pred) == int(row["gold_index"]):
            correct += 1
    return safe_div(correct, total)

def _baseline_accuracy_for_rows(rows: Sequence[Mapping[str, Any]], baseline_index: Optional[int]) -> float:
    """
    Return accuracy for the most frequent baseline over ``rows``.

    :param rows: Iterable of evaluation rows or metrics to analyse.

    :type rows: Sequence[Mapping[str, Any]]

    :param baseline_index: Precomputed index that produces baseline recommendations.

    :type baseline_index: Optional[int]

    :returns: accuracy for the most frequent baseline over ``rows``

    :rtype: float

    """
    if baseline_index is None:
        return 0.0
    if not rows:
        return 0.0
    correct = 0
    total = 0
    for row in rows:
        if not row.get("eligible"):
            continue
        total += 1
        if int(row.get("gold_index", -1)) == int(baseline_index):
            correct += 1
    return safe_div(correct, total)

def _bootstrap_uncertainty(
    *,
    rows: Sequence[Mapping[str, Any]],
    best_k: int,
    baseline_index: Optional[int],
    replicates: int,
    seed: int,
) -> Optional[Dict[str, Any]]:
    """
    Return bootstrap-based uncertainty estimates for accuracy metrics.

    :param rows: Iterable of evaluation rows or metrics to analyse.

    :type rows: Sequence[Mapping[str, Any]]

    :param best_k: Neighbourhood size selected as optimal for the evaluation.

    :type best_k: int

    :param baseline_index: Precomputed index that produces baseline recommendations.

    :type baseline_index: Optional[int]

    :param replicates: Number of bootstrap replicates to sample.

    :type replicates: int

    :param seed: Seed used to initialise pseudo-random operations.

    :type seed: int

    :returns: bootstrap-based uncertainty estimates for accuracy metrics

    :rtype: Optional[Dict[str, Any]]

    """
    if replicates <= 0:
        return None
    eligible_rows = [row for row in rows if row.get("eligible")]
    if not eligible_rows:
        return None
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for idx, row in enumerate(eligible_rows):
        group_key = row.get("group_key") or _group_key_for_example(row, idx)
        grouped.setdefault(group_key, []).append(row)
    if len(grouped) < 2:
        return None

    keys = list(grouped.keys())
    rng = default_rng(seed)
    model_samples: List[float] = []
    baseline_samples: List[float] = []
    for _ in range(replicates):
        sampled_rows: List[Mapping[str, Any]] = []
        sampled_indices = rng.integers(0, len(keys), size=len(keys))
        for key_idx in sampled_indices:
            sampled_rows.extend(grouped[keys[key_idx]])
        model_samples.append(_accuracy_for_rows(sampled_rows, best_k))
        if baseline_index is not None:
            baseline_samples.append(_baseline_accuracy_for_rows(sampled_rows, baseline_index))

    model_ci = {
        "low": float(np.percentile(model_samples, 2.5)),
        "high": float(np.percentile(model_samples, 97.5)),
    }
    model_mean = float(np.mean(model_samples))
    result: Dict[str, Any] = {
        "method": "participant_bootstrap",
        "n_groups": len(grouped),
        "n_rows": len(eligible_rows),
        "n_bootstrap": replicates,
        "seed": seed,
        "model": {
            "mean": model_mean,
            "ci95": model_ci,
        },
    }
    if baseline_samples:
        baseline_ci = {
            "low": float(np.percentile(baseline_samples, 2.5)),
            "high": float(np.percentile(baseline_samples, 97.5)),
        }
        result["baseline"] = {
            "mean": float(np.mean(baseline_samples)),
            "ci95": baseline_ci,
        }
    return result

def parse_k_values(k_default: int, sweep: str) -> List[int]:
    """
        Derive the sorted set of ``k`` values requested for evaluation.

        Parameters

        ----------

        k_default:

            The baseline ``k`` value supplied via the CLI (used when the sweep is empty).

        sweep:

            Comma-delimited string of additional ``k`` candidates provided by the user.

        Returns

        -------

        list[int]

            Strictly positive ``k`` values in ascending order. Falls back to ``k_default``

            (or ``25`` when unset) if the sweep does not contain any valid integers.

    :param k_default: Fallback ``k`` value applied when no sweep result is available.

    :type k_default: int

    :param sweep: Sweep configuration or CLI payload under inspection.

    :type sweep: str

    :returns: Sorted list of unique ``k`` values parsed from the sweep specification.

    :rtype: List[int]

    """
    values = {int(k_default)} if k_default else set()
    for token in sweep.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            values.add(int(token))
        except ValueError:
            continue
    k_vals = sorted(k for k in values if k > 0)
    return k_vals or [int(k_default) if k_default else 25]

def select_best_k(k_values: Sequence[int], accuracy_by_k: Dict[int, float]) -> int:
    """
        Choose an appropriate ``k`` by applying a simple elbow heuristic.

        Parameters

        ----------

        k_values:

            Sorted sequence of evaluated ``k`` values.

        accuracy_by_k:

            Observed accuracy for each ``k`` on the validation split.

        Returns

        -------

        int

            The ``k`` value where marginal gains fall below half of the initial slope,

            or the accuracy-maximising ``k`` when the heuristic cannot be applied.

    :param k_values: Iterable of ``k`` values to evaluate or report.

    :type k_values: Sequence[int]

    :param accuracy_by_k: Mapping from each ``k`` to its measured validation accuracy.

    :type accuracy_by_k: Dict[int, float]

    :returns: Neighbourhood size that maximises the provided accuracy scores.

    :rtype: int

    """
    if len(k_values) <= 2:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    accuracies = [accuracy_by_k.get(k, 0.0) for k in k_values]
    slopes: List[float] = []
    for idx in range(1, len(k_values)):
        delta_acc = accuracies[idx] - accuracies[idx - 1]
        delta_k = k_values[idx] - k_values[idx - 1]
        slopes.append(delta_acc / delta_k if delta_k else 0.0)
    if not slopes:
        return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))
    first_slope = slopes[0]
    threshold = max(first_slope * 0.5, 0.001)
    for idx, slope in enumerate(slopes[1:], start=1):
        if slope <= threshold:
            return k_values[idx]
    return max(k_values, key=lambda k: accuracy_by_k.get(k, 0.0))

def resolve_reports_dir(out_dir: Path) -> Path:
    """
        Resolve the canonical reports directory corresponding to ``out_dir``.

        Parameters

        ----------

        out_dir:

            Directory containing pipeline artefacts (typically under ``models/knn``).

        Returns

        -------

        pathlib.Path

            Path pointing at the root ``reports`` directory for the repository.

    :param out_dir: Output directory receiving generated metrics and reports.

    :type out_dir: Path

    :returns: Directory path where the generated report files will be stored.

    :rtype: Path

    """
    resolved = out_dir.resolve()
    parents = list(resolved.parents)
    if len(parents) >= 1 and parents[0].name == "knn":
        resolved = parents[0]
        parents = list(resolved.parents)
    if len(parents) >= 1 and parents[0].name == "models":
        root_dir = parents[0].parent
    elif len(parents) >= 2 and parents[1].name == "models":
        root_dir = parents[1].parent
    else:
        root_dir = resolved.parent
    return root_dir / "reports"

def plot_elbow(
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    best_k: int,
    output_path: Path,
    *,
    data_split: str = "validation",
) -> None:
    """
        Generate an error-rate plot to visualise the KNN elbow heuristic.

        Parameters

        ----------

        k_values:

            Iterable of evaluated ``k`` values.

        accuracy_by_k:

            Mapping from ``k`` to accuracy on the selected split.

        best_k:

            The configuration chosen for downstream reporting.

        output_path:

            Destination filename for the generated PNG.

        data_split:

            Human-readable label describing the evaluation split (defaults to ``validation``).

    :param k_values: Iterable of ``k`` values to evaluate or report.

    :type k_values: Sequence[int]

    :param accuracy_by_k: Mapping from each ``k`` to its measured validation accuracy.

    :type accuracy_by_k: Dict[int, float]

    :param best_k: Neighbourhood size selected as optimal for the evaluation.

    :type best_k: int

    :param output_path: Filesystem path for the generated report or figure.

    :type output_path: Path

    :param data_split: Name of the dataset split from which metrics were derived.

    :type data_split: str

    :returns: None.

    :rtype: None

    """
    if plt is None:
        logging.warning("[KNN] Skipping elbow plot (matplotlib not installed)")
        return

    if not k_values:
        logging.warning("[KNN] Skipping elbow plot (no k values supplied)")
        return

    plt.figure(figsize=(6, 4))
    error_rates = [1.0 - float(accuracy_by_k.get(k, 0.0)) for k in k_values]
    plt.plot(k_values, error_rates, marker="o", label="Error rate")
    if best_k in accuracy_by_k:
        best_error = 1.0 - float(accuracy_by_k[best_k])
        plt.axvline(best_k, color="red", linestyle="--", alpha=0.6)
        plt.scatter([best_k], [best_error], color="red", label="Selected k")
    split_label = data_split.strip() or "validation"
    plt.title(f"KNN error vs k ({split_label} split)")
    plt.xlabel("k")
    plt.ylabel("Error rate")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(handles, labels)
    plt.figtext(
        0.5,
        -0.05,
        f"Error computed on {split_label} data (eligible examples only)",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def compute_auc_from_curve(k_values: Sequence[int], accuracy_by_k: Dict[int, float]) -> tuple[float, float]:
    """
        Compute the area under the accuracy-vs-``k`` curve.

        Parameters

        ----------

        k_values:

            Iterable of evaluated ``k`` values.

        accuracy_by_k:

            Mapping from ``k`` to measured accuracy.

        Returns

        -------

        tuple[float, float]

            ``(auc_area, auc_normalized)`` where ``auc_area`` is the trapezoidal

            integral and ``auc_normalized`` scales the area by the extent of ``k``.

    :param k_values: Iterable of ``k`` values to evaluate or report.

    :type k_values: Sequence[int]

    :param accuracy_by_k: Mapping from each ``k`` to its measured validation accuracy.

    :type accuracy_by_k: Dict[int, float]

    :returns: Area-under-curve value computed from the accuracy trajectory.

    :rtype: tuple[float, float]

    """
    if not k_values:
        return 0.0, 0.0
    sorted_k = sorted({int(k) for k in k_values})
    accuracy_values = [float(accuracy_by_k.get(k, 0.0)) for k in sorted_k]
    if len(sorted_k) == 1:
        value = accuracy_values[0]
        return value, value
    area = float(np.trapz(accuracy_values, sorted_k))
    span = float(sorted_k[-1] - sorted_k[0]) or 1.0
    return area, area / span

def _normalise_feature_space(feature_space: str | None) -> str:
    """Return the validated feature space identifier.

    :param feature_space: Raw feature-space string supplied via CLI.
    :returns: Lowercase feature-space token (``tfidf`` or ``word2vec``).
    :raises ValueError: If an unsupported feature space is supplied.
    """
    value = (feature_space or "tfidf").lower()
    if value not in {"tfidf", "word2vec", "sentence_transformer"}:
        raise ValueError(f"Unsupported feature space '{feature_space}'")
    return value

def _word2vec_config_from_args(args, issue_slug: str) -> Word2VecConfig:
    """Return the Word2Vec configuration derived from CLI arguments.

    :param args: Parsed CLI namespace containing Word2Vec options.
    :param issue_slug: Current issue being processed (used to namespace models).
    :returns: Populated :class:`~knn.features.Word2VecConfig` instance.
    """
    default_cfg = Word2VecConfig()
    model_root = Path(args.word2vec_model_dir) if args.word2vec_model_dir else default_cfg.model_dir
    return Word2VecConfig(
        vector_size=int(args.word2vec_size),
        window=int(getattr(args, "word2vec_window", default_cfg.window)),
        min_count=int(getattr(args, "word2vec_min_count", default_cfg.min_count)),
        epochs=int(getattr(args, "word2vec_epochs", default_cfg.epochs)),
        model_dir=Path(model_root) / issue_slug,
        seed=int(getattr(args, "knn_seed", default_cfg.seed)),
        workers=int(getattr(args, "word2vec_workers", default_cfg.workers)),
    )

def _sentence_transformer_config_from_args(args) -> SentenceTransformerConfig:
    """
    Return the SentenceTransformer configuration derived from CLI arguments.

    :param args: Namespace object containing parsed command-line arguments.

    :type args: Any

    :returns: the SentenceTransformer configuration derived from CLI arguments

    :rtype: SentenceTransformerConfig

    """
    device_raw = getattr(args, "sentence_transformer_device", "")
    device = device_raw if device_raw else None
    return SentenceTransformerConfig(
        model_name=getattr(args, "sentence_transformer_model", SentenceTransformerConfig().model_name),
        device=device,
        batch_size=int(getattr(args, "sentence_transformer_batch_size", 32)),
        normalize=bool(getattr(args, "sentence_transformer_normalize", True)),
    )

def _fit_index_for_issue(
    *,
    feature_space: str,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
):
    """Build an index for the requested feature space and handle persistence.

    :param feature_space: ``tfidf`` or ``word2vec``.
    :param train_ds: Training split (Hugging Face dataset slice).
    :param issue_slug: Normalised issue identifier.
    :param extra_fields: Optional text fields to concatenate into documents.
    :param args: CLI namespace for additional parameters.
    :returns: Dictionary describing the fitted index artifacts.
    :raises ValueError: If the requested feature space is unsupported.
    """
    if feature_space == "tfidf":
        logging.info("[KNN] Building TF-IDF index for issue=%s", issue_slug)
        index = build_tfidf_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            max_features=None,
            extra_fields=extra_fields,
        )
        if args.save_index:
            save_tfidf_index(index, Path(args.save_index) / issue_slug)
        return index

    if feature_space == "word2vec":
        logging.info("[KNN] Building Word2Vec index for issue=%s", issue_slug)
        config = _word2vec_config_from_args(args, issue_slug)
        index = build_word2vec_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            extra_fields=extra_fields,
            config=config,
        )
        if args.save_index:
            save_word2vec_index(index, Path(args.save_index) / issue_slug)
        return index

    if feature_space == "sentence_transformer":
        logging.info("[KNN] Building SentenceTransformer index for issue=%s", issue_slug)
        config = _sentence_transformer_config_from_args(args)
        index = build_sentence_transformer_index(
            train_ds,
            max_train=args.knn_max_train,
            seed=args.knn_seed,
            extra_fields=extra_fields,
            config=config,
        )
        if args.save_index:
            save_sentence_transformer_index(index, Path(args.save_index) / issue_slug)
        return index

    raise ValueError(f"Unsupported feature space '{feature_space}'")

def _load_index_for_issue(
    *,
    feature_space: str,
    issue_slug: str,
    args,
):
    """Load a persisted index for the requested feature space.

    :param feature_space: ``tfidf`` or ``word2vec``.
    :param issue_slug: Normalised issue identifier.
    :param args: CLI namespace providing the ``--load-index`` directory.
    :returns: Dictionary with the loaded index artifacts.
    :raises ValueError: If the feature space is not recognised.
    """
    load_path = Path(args.load_index) / issue_slug
    if feature_space == "tfidf":
        logging.info("[KNN] Loading TF-IDF index for issue=%s", issue_slug)
        return load_tfidf_index(load_path)
    if feature_space == "word2vec":
        logging.info("[KNN] Loading Word2Vec index for issue=%s", issue_slug)
        return load_word2vec_index(load_path)
    if feature_space == "sentence_transformer":
        logging.info("[KNN] Loading SentenceTransformer index for issue=%s", issue_slug)
        return load_sentence_transformer_index(load_path)
    raise ValueError(f"Unsupported feature space '{feature_space}'")

def _build_or_load_index(
    *,
    train_ds,
    issue_slug: str,
    extra_fields: Sequence[str],
    args,
):
    """Return the KNN index for ``issue_slug`` based on CLI arguments.

    :param train_ds: Training split dataset.
    :param issue_slug: Normalised issue identifier.
    :param extra_fields: Optional extra text fields.
    :param args: CLI namespace containing ``--fit-index`` or ``--load-index``.
    :returns: Dictionary describing the fitted or loaded KNN index.
    :raises ValueError: When neither ``--fit-index`` nor ``--load-index`` is used.
    """
    feature_space = _normalise_feature_space(getattr(args, "feature_space", None))
    if args.fit_index:
        return _fit_index_for_issue(
            feature_space=feature_space,
            train_ds=train_ds,
            issue_slug=issue_slug,
            extra_fields=extra_fields,
            args=args,
        )
    if args.load_index:
        return _load_index_for_issue(
            feature_space=feature_space,
            issue_slug=issue_slug,
            args=args,
        )
    raise ValueError("Set either --fit_index or --load_index to obtain a KNN index")

def run_eval(args) -> None:  # pylint: disable=too-many-locals
    """
        Evaluate the KNN baseline across the issues specified on the CLI.

        Parameters

        ----------

        args:

            Namespace returned by :func:`knn.cli.build_parser`, including dataset,

            feature-space, and sweep configuration.

    :param args: Namespace object containing parsed command-line arguments.

    :type args: Any

    :returns: None.

    :rtype: None

    """
    dataset_source, base_ds, available_issues = prepare_dataset(
        dataset=getattr(args, "dataset", None),
        default_source=DEFAULT_DATASET_SOURCE,
        cache_dir=args.cache_dir,
        loader=load_dataset_source,
        issue_lookup=issues_in_dataset,
    )

    issue_lookup = {issue.lower(): issue for issue in available_issues}

    def _resolve_issue_list(tokens: List[str], fallback: Sequence[str]) -> List[str]:
        """
        Normalise user-provided issue tokens against available issues.

        :param tokens: Raw issue tokens supplied via CLI flags.
        :type tokens: List[str]
        :param fallback: Issues to fall back to when the token list is empty.
        :type fallback: Sequence[str]
        :returns: List of resolved issue identifiers ready for evaluation.
        :rtype: List[str]
        """
        if not tokens:
            return list(fallback)
        if any(token.lower() == "all" for token in tokens):
            return list(available_issues)
        resolved: List[str] = []
        for token in tokens:
            lookup = issue_lookup.get(token.lower())
            if lookup:
                resolved.append(lookup)
            else:
                logging.warning("[KNN] Unknown issue token '%s'; skipping.", token)
        return resolved or list(fallback)

    joint_issue_tokens = _split_tokens(getattr(args, "issues", ""))
    eval_issue_tokens = _split_tokens(getattr(args, "eval_issues", "")) or joint_issue_tokens
    train_issue_tokens = _split_tokens(getattr(args, "train_issues", ""))

    issues = _resolve_issue_list(eval_issue_tokens, available_issues)

    joint_study_tokens = _split_tokens(getattr(args, "participant_studies", ""))
    train_study_tokens = _split_tokens(getattr(args, "train_participant_studies", "")) or joint_study_tokens
    eval_study_tokens = _split_tokens(getattr(args, "eval_participant_studies", "")) or joint_study_tokens

    k_values = parse_k_values(args.knn_k, args.knn_k_sweep)
    logging.info("[KNN] Evaluating k values: %s", k_values)

    base_provenance = _collect_dataset_provenance(base_ds)
    repo_state = _collect_repo_state()

    train_split = base_ds.get(TRAIN_SPLIT)
    eval_split = base_ds.get(EVAL_SPLIT)
    if train_split is None or eval_split is None:
        raise RuntimeError(f"Dataset '{dataset_source}' must expose '{TRAIN_SPLIT}' and '{EVAL_SPLIT}' splits.")

    for issue in issues:
        base_issue_slug = issue.replace(" ", "_")
        issue_slug = base_issue_slug

        current_train_issues = (
            _resolve_issue_list(train_issue_tokens, available_issues)
            if train_issue_tokens
            else ([issue] if issue != "all" else [])
        )

        filtered_train = train_split
        if current_train_issues:
            filtered_train = _filter_split_for_issues(filtered_train, current_train_issues)
        filtered_eval = eval_split
        if issue != "all":
            filtered_eval = _filter_split_for_issues(filtered_eval, [issue])

        filtered_train = filter_split_for_participant_studies(filtered_train, train_study_tokens)
        filtered_eval = filter_split_for_participant_studies(filtered_eval, eval_study_tokens)

        if len(filtered_eval) == 0:
            logging.warning(
                "[KNN] Skipping issue=%s (no evaluation rows after filters: eval_studies=%s eval_issue=%s)",
                base_issue_slug,
                ",".join(eval_study_tokens) or "all",
                issue,
            )
            continue
        if args.fit_index and len(filtered_train) == 0:
            logging.warning(
                "[KNN] Skipping issue=%s (no training rows after filters and --fit-index enabled)",
                base_issue_slug,
            )
            continue

        issue_slug = compose_issue_slug(issue, eval_study_tokens)

        active_provenance = _collect_dataset_provenance(
            {
                TRAIN_SPLIT: filtered_train,
                EVAL_SPLIT: filtered_eval,
            }
        )
        logging.info(
            "[KNN] issue=%s train_rows=%d eval_rows=%d train_studies=%s eval_studies=%s train_issues=%s",
            issue_slug,
            len(filtered_train),
            len(filtered_eval),
            ",".join(train_study_tokens) or "all",
            ",".join(eval_study_tokens) or "all",
            ",".join(current_train_issues) or ("(inherit:" + issue + ")"),
        )
        provenance = {
            "dataset": {
                "source": dataset_source,
                "base": base_provenance,
                "active": active_provenance,
                "filters": {
                    "train_issues": current_train_issues,
                    "eval_issue": issue,
                    "train_participant_studies": list(train_study_tokens),
                    "eval_participant_studies": list(eval_study_tokens),
                },
            },
            "code": repo_state,
        }

        extra_fields = [
            token.strip()
            for token in (args.knn_text_fields or "").split(",")
            if token.strip()
        ]
        knn_index = _build_or_load_index(
            train_ds=filtered_train,
            issue_slug=issue_slug,
            extra_fields=extra_fields,
            args=args,
        )
        feature_space = str(knn_index.get("feature_space", "tfidf")).lower()

        evaluate_issue(
            issue_slug=issue_slug,
            dataset_source=dataset_source,
            train_ds=filtered_train if len(filtered_train) else None,
            eval_ds=filtered_eval,
            k_values=k_values,
            knn_index=knn_index,
            extra_fields=extra_fields,
            feature_space=feature_space,
            args=args,
            provenance=provenance,
        )

# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements

def _write_issue_outputs(
    *,
    args,
    issue_slug: str,
    feature_space: str,
    dataset_source: str,
    rows: Sequence[Dict[str, Any]],
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    best_k: int,
    bucket_stats: Dict[str, Dict[str, int]],
    single_multi_stats: Dict[str, int],
    gold_hist: Dict[int, int],
    per_k_stats: Dict[int, Dict[str, int]],
    extra_fields: Sequence[str],
    curve_metrics: Dict[str, Any],
    provenance: Mapping[str, Any],
    uncertainty: Optional[Mapping[str, Any]],
) -> None:
    """Persist evaluation artifacts and per-``k`` directories for an issue.

    :param args: CLI namespace controlling output directories.
    :param issue_slug: Issue slug associated with the evaluation batch.
    :param feature_space: Active feature space (``tfidf`` or ``word2vec``).
    :param dataset_source: Source dataset label written to metrics.
    :param rows: Per-example records produced during evaluation.
    :param k_values: Sequence of ``k`` values scored for the issue.
    :param accuracy_by_k: Accuracy measured for each ``k``.
    :param best_k: Elbow-selected ``k`` value.
    :param bucket_stats: Aggregated slate-position statistics.
    :param single_multi_stats: Aggregated single vs multi option metrics.
    :param gold_hist: Histogram of gold indices encountered.
    :param per_k_stats: Eligibility and correctness tallies per ``k``.
    :param extra_fields: Extra text fields contributing to the query document.
    :param curve_metrics: Serialised evaluation/train curve diagnostics.
    """
    best_accuracy = accuracy_by_k.get(best_k, 0.0)
    eligible_overall = int(per_k_stats.get(best_k, {}).get("eligible", 0))

    issue_dir = Path(args.out_dir) / issue_slug
    issue_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = issue_dir / f"knn_eval_{issue_slug}_{EVAL_SPLIT}.jsonl"
    metrics_json = issue_dir / f"knn_eval_{issue_slug}_{EVAL_SPLIT}_metrics.json"

    with open(out_jsonl, "w", encoding="utf-8") as handle:
        for row in rows:
            preds_serializable = {
                str(k): (int(v) if v is not None else None)
                for k, v in row["predictions_by_k"].items()
            }
            best_pred = row["predictions_by_k"].get(best_k)
            record = {
                "knn_pred_index": int(best_pred) if best_pred is not None else None,
                "gold_index": row["gold_index"],
                "n_options": row["n_options"],
                "correct": bool(best_pred is not None and int(best_pred) == row["gold_index"]),
                "eligible": row["eligible"],
                "position_index": row["position_index"],
                "position_bucket": row["position_bucket"],
                "issue": issue_slug if issue_slug != "all" else row.get("issue_value"),
                "predictions_by_k": preds_serializable,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    pos_stats_out = {
        bucket: int(bucket_stats["position_seen"][bucket])
        for bucket in BUCKET_LABELS
    }
    accuracy_opts_out = {
        bucket: safe_div(
            bucket_stats["options_correct"][bucket],
            bucket_stats["options_eligible"][bucket],
        )
        for bucket in BUCKET_LABELS
    }

    gold_distribution = {str(k): int(v) for k, v in sorted(gold_hist.items())}
    if gold_hist and eligible_overall:
        top_idx, top_count = max(gold_hist.items(), key=lambda kv: kv[1])
        baseline_accuracy = safe_div(top_count, eligible_overall)
    else:
        top_idx = None
        baseline_accuracy = 0.0

    random_sum = float(single_multi_stats.get("rand_inverse_sum", 0.0))
    random_count = int(single_multi_stats.get("rand_inverse_count", 0))
    random_baseline = safe_div(random_sum, random_count)
    accuracy_by_k_serializable = {
        str(k): float(accuracy_by_k[k])
        for k in k_values
    }

    reports_dir = resolve_reports_dir(Path(args.out_dir)) / "knn" / feature_space
    reports_dir.mkdir(parents=True, exist_ok=True)
    elbow_path = reports_dir / f"elbow_{issue_slug}.png"
    plot_elbow(k_values, accuracy_by_k, best_k, elbow_path, data_split=EVAL_SPLIT)

    curve_json = issue_dir / f"knn_curves_{issue_slug}.json"
    with open(curve_json, "w", encoding="utf-8") as handle:
        json.dump(curve_metrics, handle, ensure_ascii=False, indent=2)

    metrics: Dict[str, Any] = {
        "model": "knn",
        "feature_space": feature_space,
        "dataset": dataset_source,
        "issue": issue_slug,
        "split": EVAL_SPLIT,
        "n_total": int(len(rows)),
        "n_eligible": int(eligible_overall),
        "accuracy_overall": best_accuracy,
        "accuracy_by_k": accuracy_by_k_serializable,
        "best_k": int(best_k),
        "position_stats": pos_stats_out,
        "by_n_options": {
            bucket: {
                "hist_seen": int(bucket_stats["options_seen"][bucket]),
                "hist_eligible": int(bucket_stats["options_eligible"][bucket]),
                "hist_correct": int(bucket_stats["options_correct"][bucket]),
                "accuracy": accuracy_opts_out[bucket],
            }
            for bucket in BUCKET_LABELS
        },
        "split_single_vs_multi": {
            "n_single": int(single_multi_stats["seen_single"]),
            "n_multi": int(single_multi_stats["seen_multi"]),
            "eligible_single": int(single_multi_stats["elig_single"]),
            "eligible_multi": int(single_multi_stats["elig_multi"]),
            "accuracy_single": safe_div(
                single_multi_stats["corr_single"],
                single_multi_stats["elig_single"],
            ),
            "accuracy_multi": safe_div(
                single_multi_stats["corr_multi"],
                single_multi_stats["elig_multi"],
            ),
        },
        "gold_index_distribution": gold_distribution,
        "baseline_most_frequent_gold_index": {
            "top_index": top_idx,
            "count": int(gold_hist.get(top_idx, 0) if top_idx is not None else 0),
            "accuracy": baseline_accuracy,
        },
        "random_baseline_expected_accuracy": random_baseline,
        "knn_hparams": {
            "k": int(args.knn_k),
            "k_sweep": [int(k) for k in k_values],
            "metric": args.knn_metric,
            "fit_index": bool(args.fit_index),
            "save_index": args.save_index or "",
            "load_index": args.load_index or "",
            "text_fields": list(extra_fields),
        },
        "elbow_plot": str(elbow_path),
        "per_k_directories": {
            str(k): str((issue_dir / f"k-{int(k)}"))
            for k in k_values
        },
        "curve_metrics": curve_metrics,
        "curve_metrics_path": str(curve_json),
        "notes": "Accuracy computed over eligible rows (gold_index>0).",
    }
    if uncertainty:
        model_uncertainty = uncertainty.get("model")
        baseline_uncertainty = uncertainty.get("baseline")
        if model_uncertainty:
            metrics["accuracy_ci_95"] = model_uncertainty.get("ci95")
            metrics["accuracy_uncertainty"] = model_uncertainty
        if baseline_uncertainty:
            metrics["baseline_ci_95"] = baseline_uncertainty.get("ci95")
            metrics["baseline_uncertainty"] = baseline_uncertainty
        metrics["uncertainty"] = uncertainty
    metrics["provenance"] = provenance

    with open(metrics_json, "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    for k in k_values:
        k_int = int(k)
        k_dir = issue_dir / f"k-{k_int}"
        k_dir.mkdir(parents=True, exist_ok=True)
        k_predictions_path = k_dir / f"predictions_{issue_slug}_{EVAL_SPLIT}.jsonl"
        with open(k_predictions_path, "w", encoding="utf-8") as handle:
            for row in rows:
                pred_value = row["predictions_by_k"].get(k_int)
                record = {
                    "k": k_int,
                    "knn_pred_index": int(pred_value) if pred_value is not None else None,
                    "gold_index": row["gold_index"],
                    "eligible": row["eligible"],
                    "correct": bool(pred_value is not None and int(pred_value) == row["gold_index"]),
                    "n_options": row["n_options"],
                    "position_index": row["position_index"],
                    "issue": issue_slug if issue_slug != "all" else row.get("issue_value"),
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        k_stats = per_k_stats[k_int]
        k_metrics: Dict[str, Any] = {
            "model": "knn",
            "feature_space": feature_space,
            "dataset": dataset_source,
            "issue": issue_slug,
            "split": EVAL_SPLIT,
            "k": k_int,
            "n_total": int(len(rows)),
            "n_eligible": int(k_stats["eligible"]),
            "n_correct": int(k_stats["correct"]),
            "accuracy": float(accuracy_by_k.get(k_int, 0.0)),
            "elbow_plot": str(elbow_path),
        }
        if uncertainty:
            k_metrics["uncertainty"] = uncertainty
        k_metrics["provenance"] = provenance
        with open(k_dir / f"metrics_{issue_slug}_{EVAL_SPLIT}.json", "w", encoding="utf-8") as handle:
            json.dump(k_metrics, handle, ensure_ascii=False, indent=2)

    logging.info(
        "[DONE][%s] split=%s n=%d eligible=%d knn_acc=%.4f (best_k=%d)",
        issue_slug,
        EVAL_SPLIT,
        len(rows),
        eligible_overall,
        best_accuracy,
        best_k,
    )
    logging.info("[WROTE] per-example: %s", out_jsonl)
    logging.info("[WROTE] metrics: %s", metrics_json)
    logging.info("[WROTE] curves: %s", curve_json)
def _accumulate_row(
    *,
    example: Dict[str, Any],
    bucket_stats: Dict[str, Dict[str, int]],
    per_k_stats: Dict[int, Dict[str, int]],
    single_multi_stats: Dict[str, int],
    gold_hist: Dict[int, int],
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    query_config: SlateQueryConfig,
    row_index: int,
) -> Dict[str, Any]:
    """Process a single evaluation example and update aggregate statistics.

    :param example: Dataset row representing one recommendation slate.
    :param bucket_stats: Mutable dictionary tracking per-bucket counts.
    :param per_k_stats: Mutable dictionary storing eligible/correct tallies per ``k``.
    :param single_multi_stats: Mutable dictionary tracking single vs multi-option stats.
    :param gold_hist: Mutable histogram of observed gold indices.
    :param k_values: Sequence of ``k`` values to score.
    :param knn_index: Prepared KNN index artifacts.
    :param query_config: Configuration controlling query generation.
    :returns: Serialised per-example record including predictions for each ``k``.
    """
    slate_pairs = extract_slate_items(example)
    n_options = len(slate_pairs)
    n_bucket = bin_nopts(n_options)
    bucket_stats["options_seen"][n_bucket] += 1

    try:
        position = int(example.get("video_index") or -1)
    except (TypeError, ValueError):
        position = -1
    pos_bucket = bucket_from_pos(position)
    bucket_stats["position_seen"][pos_bucket] += 1

    gold_index = int(example.get("gold_index") or -1)
    gold_raw = str(example.get(SOLUTION_COLUMN, "")).strip()
    if gold_index < 1 and slate_pairs:
        for option_index, (title, vid) in enumerate(slate_pairs, start=1):
            if gold_raw and (gold_raw == vid or canon(gold_raw) == canon(title)):
                gold_index = option_index
                break

    predictions = knn_predict_among_slate_multi(
        knn_index=knn_index,
        example=example,
        k_values=k_values,
        config=query_config,
    )

    eligible = gold_index > 0 and n_options > 0
    if eligible:
        bucket_stats["options_eligible"][n_bucket] += 1
        gold_hist[gold_index] = gold_hist.get(gold_index, 0) + 1
        single_multi_stats["rand_inverse_sum"] += (1.0 / n_options) if n_options else 0.0
        single_multi_stats["rand_inverse_count"] += 1
        if n_options == 1:
            single_multi_stats["elig_single"] += 1
        else:
            single_multi_stats["elig_multi"] += 1
    if n_options == 1:
        single_multi_stats["seen_single"] += 1
    elif n_options > 1:
        single_multi_stats["seen_multi"] += 1

    if eligible:
        for k, pred in predictions.items():
            k_stats = per_k_stats[k]
            k_stats["eligible"] += 1
            if pred is not None and int(pred) == gold_index:
                k_stats["correct"] += 1

    group_key = _group_key_for_example(example, row_index)

    return {
        "predictions_by_k": predictions,
        "gold_index": int(gold_index),
        "n_options": int(n_options),
        "n_options_bucket": n_bucket,
        "eligible": bool(eligible),
        "position_index": int(position),
        "position_bucket": pos_bucket,
        "issue_value": example.get("issue"),
        "group_key": group_key,
    }

def _evaluate_dataset_split(
    *,
    dataset,
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    extra_fields: Sequence[str],
    metric: str,
    capture_rows: bool,
    log_label: str,
    max_examples: int | None,
    log_k: int | None = None,
) -> Dict[str, Any]:
    """Return aggregate statistics for ``dataset`` using the provided index.

    :param dataset: Dataset slice to iterate.
    :param k_values: Sequence of ``k`` values to evaluate.
    :param knn_index: Prepared KNN index artifacts.
    :param extra_fields: Extra text fields appended to the query.
    :param metric: Distance metric for candidate scoring (``l2`` or ``cosine``).
    :param capture_rows: Whether to retain per-example prediction rows.
    :param log_label: Label emitted in progress logs.
    :param max_examples: Optional limit on the number of processed examples.
    :param log_k: Optional ``k`` to report accuracy for in progress logs.
    :returns: Dictionary containing rows, aggregate stats, and counts.
    """
    rows: List[Dict[str, Any]] = [] if capture_rows else []
    gold_hist: Dict[int, int] = {}
    bucket_stats = {
        "position_seen": {b: 0 for b in BUCKET_LABELS},
        "options_seen": {b: 0 for b in BUCKET_LABELS},
        "options_eligible": {b: 0 for b in BUCKET_LABELS},
        "options_correct": {b: 0 for b in BUCKET_LABELS},
    }
    per_k_stats = {k: {"eligible": 0, "correct": 0} for k in k_values}
    single_multi_stats: Dict[str, float | int] = {
        "seen_single": 0,
        "seen_multi": 0,
        "elig_single": 0,
        "elig_multi": 0,
        "corr_single": 0,
        "corr_multi": 0,
        "rand_inverse_sum": 0.0,
        "rand_inverse_count": 0,
    }

    dataset_len = len(dataset)
    limit = dataset_len
    if max_examples is not None and max_examples > 0:
        limit = min(dataset_len, max_examples)

    query_config = SlateQueryConfig(
        text_fields=tuple(extra_fields),
        lowercase=True,
        metric=metric,
    )

    log_k_value: int | None = None
    if log_k:
        desired = int(log_k)
        if desired in per_k_stats:
            log_k_value = desired
        elif per_k_stats:
            log_k_value = min(per_k_stats.keys(), key=lambda key_k: abs(key_k - desired))

    start_time = time.time()

    for idx in range(int(limit)):
        row = _accumulate_row(
            example=dataset[int(idx)],
            bucket_stats=bucket_stats,
            per_k_stats=per_k_stats,
            single_multi_stats=single_multi_stats,  # type: ignore[arg-type]
            gold_hist=gold_hist,
            k_values=k_values,
            knn_index=knn_index,
            query_config=query_config,
            row_index=int(idx),
        )
        if capture_rows:
            rows.append(row)
        if (idx + 1) % 25 == 0:
            elapsed = time.time() - start_time
            acc_message = ""
            if log_k_value is not None:
                stats = per_k_stats[log_k_value]
                acc_message = f"  acc@{log_k_value}={safe_div(stats['correct'], stats['eligible']):.3f}"
            logging.info(
                "[%s] %d/%d  elapsed=%.1fs%s",
                log_label,
                idx + 1,
                limit,
                elapsed,
                acc_message,
            )

    return {
        "rows": rows,
        "bucket_stats": bucket_stats,
        "per_k_stats": per_k_stats,
        "single_multi_stats": single_multi_stats,
        "gold_hist": gold_hist,
        "n_examples": int(limit),
    }

def _update_correct_counts(
    rows: Sequence[Dict[str, Any]],
    best_k: int,
    bucket_stats: Dict[str, Dict[str, int]],
    single_multi_stats: Dict[str, int],
) -> None:
    """Update bucket-level correctness tallies for the selected ``best_k``.

    :param rows: Iterable of per-example prediction records.
    :param best_k: Elbow-selected ``k`` used to judge correctness.
    :param bucket_stats: Mutable dictionary storing per-bucket correctness.
    :param single_multi_stats: Mutable dictionary tracking single vs multi counts.
    """
    for row in rows:
        if not row["eligible"]:
            continue
        prediction = row["predictions_by_k"].get(best_k)
        if prediction is None:
            continue
        if int(prediction) == row["gold_index"]:
            bucket_stats["options_correct"][row["n_options_bucket"]] += 1
            if row["n_options"] == 1:
                single_multi_stats["corr_single"] += 1
            else:
                single_multi_stats["corr_multi"] += 1

def _curve_summary(
    *,
    k_values: Sequence[int],
    accuracy_by_k: Dict[int, float],
    per_k_stats: Dict[int, Dict[str, int]],
    best_k: int,
    n_examples: int,
) -> Dict[str, Any]:
    """
    Return a serialisable summary for accuracy-vs-k curves.

    :param k_values: Iterable of ``k`` values to evaluate or report.

    :type k_values: Sequence[int]

    :param accuracy_by_k: Mapping from each ``k`` to its measured validation accuracy.

    :type accuracy_by_k: Dict[int, float]

    :param per_k_stats: Detailed per-``k`` statistics derived from the evaluation curve.

    :type per_k_stats: Dict[int, Dict[str, int]]

    :param best_k: Neighbourhood size selected as optimal for the evaluation.

    :type best_k: int

    :param n_examples: Total number of evaluation examples summarised in the bundle.

    :type n_examples: int

    :returns: a serialisable summary for accuracy-vs-k curves

    :rtype: Dict[str, Any]

    """
    area, normalised = compute_auc_from_curve(k_values, accuracy_by_k)
    sorted_k = sorted({int(k) for k in k_values})
    accuracy_serialised = {
        str(k): float(accuracy_by_k.get(k, 0.0))
        for k in sorted_k
    }
    eligible_serialised = {
        str(k): int(per_k_stats[k]["eligible"])
        for k in sorted_k
    }
    correct_serialised = {
        str(k): int(per_k_stats[k]["correct"])
        for k in sorted_k
    }
    return {
        "accuracy_by_k": accuracy_serialised,
        "eligible_by_k": eligible_serialised,
        "correct_by_k": correct_serialised,
        "auc_area": float(area),
        "auc_normalized": float(normalised),
        "best_k": int(best_k),
        "best_accuracy": float(accuracy_by_k.get(best_k, 0.0)),
        "n_examples": int(n_examples),
    }

def evaluate_issue(
    *,
    issue_slug: str,
    dataset_source: str,
    train_ds,
    eval_ds,
    k_values: Sequence[int],
    knn_index: Dict[str, Any],
    extra_fields: Sequence[str],
    feature_space: str,
    args,
    provenance: Mapping[str, Any],
) -> None:  # pylint: disable=too-many-locals
    """
        Evaluate a single issue slice and persist metrics, curves, and predictions.

        Parameters

        ----------

        issue_slug:

            Normalised identifier for the issue under evaluation (used in paths).

        dataset_source:

            Name or path of the dataset backing the current run.

        train_ds:

            Training split (may be ``None``) used for optional curve diagnostics.

        eval_ds:

            Evaluation split containing the rows scored for reporting.

        k_values:

            Sequence of ``k`` values to consider when computing accuracy.

        knn_index:

            Prepared KNN index artefacts for the active feature space.

        extra_fields:

            Additional text fields concatenated into the query document.

        feature_space:

            Feature space identifier (``tfidf``, ``word2vec``, or ``sentence_transformer``).

        args:

            Parsed CLI namespace controlling evaluation behaviour.

        provenance:

            Mapping of provenance metadata recorded alongside the metrics.

    :param issue_slug: Slugified identifier representing the study/issue on disk.

    :type issue_slug: str

    :param dataset_source: Identifier describing where the dataset was loaded from.

    :type dataset_source: str

    :param train_ds: Training dataset split used to build the index.

    :type train_ds: Any

    :param eval_ds: Evaluation dataset split passed to the scorer.

    :type eval_ds: Any

    :param k_values: Iterable of ``k`` values to evaluate or report.

    :type k_values: Sequence[int]

    :param knn_index: Fitted KNN index used to score neighbours.

    :type knn_index: Dict[str, Any]

    :param extra_fields: Additional dataset columns included when building prompts or outputs.

    :type extra_fields: Sequence[str]

    :param feature_space: Feature space identifier such as ``tfidf`` or ``word2vec``.

    :type feature_space: str

    :param args: Namespace object containing parsed command-line arguments.

    :type args: Any

    :param provenance: Metadata describing how the evaluation artefacts were produced.

    :type provenance: Mapping[str, Any]

    :returns: None.

    :rtype: None

    """
    k_values_int = sorted({int(k) for k in k_values if int(k) > 0})
    eval_max = args.eval_max if args.eval_max and args.eval_max > 0 else None
    eval_summary = _evaluate_dataset_split(
        dataset=eval_ds,
        k_values=k_values_int,
        knn_index=knn_index,
        extra_fields=extra_fields,
        metric=args.knn_metric,
        capture_rows=True,
        log_label=f"eval][{issue_slug}",
        max_examples=eval_max,
        log_k=args.knn_k,
    )

    rows: List[Dict[str, Any]] = eval_summary["rows"]
    bucket_stats: Dict[str, Dict[str, int]] = eval_summary["bucket_stats"]
    single_multi_stats: Dict[str, int] = eval_summary["single_multi_stats"]  # type: ignore[assignment]
    gold_hist: Dict[int, int] = eval_summary["gold_hist"]
    per_k_stats: Dict[int, Dict[str, int]] = eval_summary["per_k_stats"]

    accuracy_by_k = {
        k: safe_div(per_k_stats[k]["correct"], per_k_stats[k]["eligible"])
        for k in k_values_int
    }
    best_k = select_best_k(k_values_int, accuracy_by_k)
    _update_correct_counts(rows, best_k, bucket_stats, single_multi_stats)
    eval_curve = _curve_summary(
        k_values=k_values_int,
        accuracy_by_k=accuracy_by_k,
        per_k_stats=per_k_stats,
        best_k=best_k,
        n_examples=eval_summary["n_examples"],
    )

    train_curve = None
    train_max = getattr(args, "train_curve_max", 0)
    if train_ds is not None:
        max_examples = train_max if train_max and train_max > 0 else None
        train_summary = _evaluate_dataset_split(
            dataset=train_ds,
            k_values=k_values_int,
            knn_index=knn_index,
            extra_fields=extra_fields,
            metric=args.knn_metric,
            capture_rows=False,
            log_label=f"train][{issue_slug}",
            max_examples=max_examples,
            log_k=args.knn_k,
        )
        train_accuracy_by_k = {
            k: safe_div(train_summary["per_k_stats"][k]["correct"], train_summary["per_k_stats"][k]["eligible"])
            for k in k_values_int
        }
        train_best_k = select_best_k(k_values_int, train_accuracy_by_k)
        train_curve = _curve_summary(
            k_values=k_values_int,
            accuracy_by_k=train_accuracy_by_k,
            per_k_stats=train_summary["per_k_stats"],
            best_k=train_best_k,
            n_examples=train_summary["n_examples"],
        )

    curve_metrics = {"eval": eval_curve}
    if train_curve:
        curve_metrics["train"] = train_curve

    baseline_index: Optional[int] = None
    if gold_hist:
        baseline_index = max(gold_hist.items(), key=lambda kv: kv[1])[0]

    replicates = int(getattr(args, "bootstrap_replicates", 500) or 0)
    bootstrap_seed = int(getattr(args, "bootstrap_seed", 2024) or 2024)
    uncertainty = _bootstrap_uncertainty(
        rows=rows,
        best_k=best_k,
        baseline_index=baseline_index,
        replicates=replicates,
        seed=bootstrap_seed,
    )

    _write_issue_outputs(
        args=args,
        issue_slug=issue_slug,
        feature_space=feature_space,
        dataset_source=dataset_source,
        rows=rows,
        k_values=k_values_int,
        accuracy_by_k=accuracy_by_k,
        best_k=best_k,
        bucket_stats=bucket_stats,
        single_multi_stats=single_multi_stats,
        gold_hist=gold_hist,
        per_k_stats=per_k_stats,
        extra_fields=extra_fields,
        curve_metrics=curve_metrics,
        provenance=provenance,
        uncertainty=uncertainty,
    )

def bin_nopts(option_count: int) -> str:
    """Bucket the number of slate options into reporting-friendly categories."""
    if option_count <= 1:
        return "1"
    if option_count == 2:
        return "2"
    if option_count == 3:
        return "3"
    if option_count == 4:
        return "4"
    return "5+"

def bucket_from_pos(pos_idx: int) -> str:
    """
        Bucket a 0-based position index into the standard reporting bins.

        Parameters

        ----------

        pos_idx:

            Zero-based position of the correct item within the retrieved slate.

        Returns

        -------

        str

            One of ``{"unknown", "1", "2", "3", "4", "5+"}`` describing the bucket.

    :param pos_idx: Position index being transformed into a categorical bucket.

    :type pos_idx: int

    :returns: Named bucket representing the supplied positional index.

    :rtype: str

    """
    if pos_idx < 0:
        return "unknown"
    if pos_idx == 0:
        return "1"
    if pos_idx == 1:
        return "2"
    if pos_idx == 2:
        return "3"
    if pos_idx == 3:
        return "4"
    return "5+"

def canon(text: str) -> str:
    """
        Canonicalise a text fragment by lowercasing and removing punctuation.

        Parameters

        ----------

        text:

            Raw text fragment to normalise.

        Returns

        -------

        str

            Canonical form suitable for equality comparisons.

    :param text: Free-form text string that requires normalisation.

    :type text: str

    :returns: Canonicalised text normalised for downstream processing.

    :rtype: str

    """
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower().strip())

__all__ = [
    "parse_k_values",
    "select_best_k",
    "resolve_reports_dir",
    "plot_elbow",
    "compute_auc_from_curve",
    "evaluate_issue",
    "bin_nopts",
    "bucket_from_pos",
    "canon",
    "run_eval",
]

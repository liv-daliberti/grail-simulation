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

"""Top-level portfolio report comparing model families across tasks.

This module gathers results produced by the individual pipelines (KNN/XGB/GPT-4o)
and RLHF families (GRPO/GRAIL) and materialises a summary under
``reports/main/README.md`` so it is easy to spot which modelling type performs
best on each study.

For the opinion task, the KNN and XGB entries are sourced from their
``opinion_from_next`` runs as requested.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

from common.pipeline.io import write_markdown_lines
from common.opinion import DEFAULT_SPECS
from .utils import start_markdown_report
from .tables import append_markdown_table


ModelKey = str  # One of: "gpt4o", "grpo", "grail", "knn", "xgb"
StudyLabel = str  # e.g. "Study 1 – Gun Control (MTurk)"
StudyScores = dict[StudyLabel, float]
StudyScoresPair = tuple[StudyScores, StudyScores]


@dataclass(frozen=True)
class MetricBundle:
    """Per-study metrics used when assembling portfolio tables.

    :param next_video_accuracy: Eligible accuracy for the next‑video task
        (0–1) or ``None`` when the value is unavailable.
    :param opinion_direction: Directional accuracy for opinion shift
        (0–1) or ``None`` when missing.
    :param opinion_mae: Mean absolute error for opinion predictions
        (non‑negative) or ``None`` when missing.
    """
    next_video_accuracy: Optional[float] = None
    opinion_direction: Optional[float] = None
    opinion_mae: Optional[float] = None


def _fmt_rate(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if math.isnan(v) or math.isinf(v):
        return "—"
    return f"{v:.{digits}f}"


def _fmt_int(value: Optional[int]) -> str:
    """Format an integer with thousands separators or an em dash."""
    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _wilson_ci(p: float, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Uses z≈1.96 for a 95% CI by default. Returns (lower, upper).
    """
    if n <= 0 or not (0.0 <= p <= 1.0):
        return (0.0, 0.0)
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2 * n)) / denom
    half_width = z * math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n)) / denom
    lo = max(0.0, center - half_width)
    hi = min(1.0, center + half_width)
    return (lo, hi)


def _fmt_rate_ci(p: Optional[float], n: Optional[int]) -> str:
    """Format a rate with its 95% Wilson CI in brackets when possible."""
    try:
        if p is None or n is None or n <= 0:
            return "—"
        p_f = float(p)
        lo, hi = _wilson_ci(p_f, int(n))
        return f"{_fmt_rate(p_f)} [{_fmt_rate(lo)}, {_fmt_rate(hi)}]"
    except Exception:  # pragma: no cover - defensive
        return _fmt_rate(p)


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _study_order() -> list[StudyLabel]:
    return [spec.label for spec in DEFAULT_SPECS]


def _label_by_key() -> dict[str, str]:
    return {spec.key: spec.label for spec in DEFAULT_SPECS}


# ------------------------
# Internal helpers
# ------------------------

def _is_checkpoint_50(path: Path) -> bool:
    parts = path.as_posix().split("/")
    return "checkpoint-50" in parts


def _collect_next_video_candidates(model_root: Path) -> list[Path]:
    """Collect candidate next_video metrics.json paths for a model family.

    Prefers a flat ``next_video`` layout when present, otherwise searches under
    the whole family directory. Excludes any file within a ``sweeps`` subtree.
    """
    candidates: list[Path] = []

    # First preference: flat next_video tree at the family root
    flat_root = model_root / "next_video"
    if flat_root.exists():
        for sub in sorted(flat_root.glob("*/metrics.json")):
            if "/sweeps/" in sub.as_posix():
                continue
            candidates.append(sub)
        if not candidates:
            candidates.extend(sorted(flat_root.rglob("metrics.json")))

    # Fallback: recursively search under the family root for any next_video metrics
    if not candidates and model_root.exists():
        for path in sorted(model_root.rglob("metrics.json")):
            posix = path.as_posix()
            if "/next_video/" not in posix:
                continue
            if "/sweeps/" in posix:
                continue
            candidates.append(path)

    return candidates


# ------------------------
# Next-video readers
# ------------------------

def _read_knn_next_video(repo_root: Path) -> dict[StudyLabel, float]:
    path = repo_root / "reports" / "knn" / "next_video" / "metrics.csv"
    best: dict[StudyLabel, float] = {}
    if not path.exists():
        return best
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                study = str(row.get("study") or "").strip()
                acc = float(row.get("accuracy") or "")
            except (TypeError, ValueError):
                continue
            prev = best.get(study)
            if prev is None or acc > prev:
                best[study] = acc
    return best


def _read_xgb_next_video(repo_root: Path) -> dict[StudyLabel, float]:
    path = repo_root / "reports" / "xgb" / "next_video" / "metrics.csv"
    values: dict[StudyLabel, float] = {}
    if not path.exists():
        return values
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study = str(row.get("study") or "").strip()
            try:
                # Eligible-only accuracy aligns with KNN reporting
                acc = float(row.get("accuracy_eligible") or row.get("accuracy") or "")
            except (TypeError, ValueError):
                continue
            values[study] = acc
    return values


def _read_grouped_json_next_video(repo_root: Path, family: str) -> dict[StudyLabel, float]:
    """Return per-study accuracy from JSON metrics for next-video.

    Supports both flat layouts (e.g. ``models/gpt-4o/next_video/<label>/metrics.json``)
    and nested RLHF layouts (e.g.
    ``models/<family>/<scenario>/checkpoint-50/next_video/**/metrics.json``).
    Aggregates across multiple runs so studies split by issue are combined.
    """
    model_root = repo_root / "models" / family
    label_map = _label_by_key()
    values: dict[StudyLabel, float] = {}
    if not model_root.exists():
        return values

    candidates = _collect_next_video_candidates(model_root)

    if not candidates:
        return values

    # Process non-checkpoint candidates first, then checkpoint-50 variants so they override
    ordered: list[Path] = sorted(
        candidates,
        key=lambda path_obj: (
            0 if not _is_checkpoint_50(path_obj) else 1,
            path_obj.as_posix(),
        ),
    )

    for metrics_path in ordered:
        payload = _load_json(metrics_path)
        group = payload.get("group_metrics", {}) if isinstance(payload, Mapping) else {}
        by_study = group.get("by_participant_study", {}) if isinstance(group, Mapping) else {}
        if not isinstance(by_study, Mapping):
            continue
        for key, study_payload in by_study.items():
            if not isinstance(study_payload, Mapping):
                continue
            label = label_map.get(str(key))
            if not label:
                continue
            try:
                acc = float(study_payload.get("accuracy"))
            except (TypeError, ValueError):
                continue
            values[label] = acc
    return values


def _gather_next_video_baselines(repo_root: Path) -> tuple[dict[StudyLabel, float], dict[StudyLabel, float], dict[StudyLabel, int]]:
    """Collect per-study next-video baselines and eligible counts.

    Returns three maps keyed by study label:
      - random baseline expected accuracy
      - most-frequent gold-index baseline accuracy
      - eligible row counts (N)
    """
    label_map = _label_by_key()
    random_map: dict[StudyLabel, float] = {}
    mostfreq_map: dict[StudyLabel, float] = {}
    eligible_map: dict[StudyLabel, int] = {}

    def _apply_from_metrics(metrics_path: Path) -> None:
        payload = _load_json(metrics_path)
        if not isinstance(payload, Mapping):
            return
        group = payload.get("group_metrics", {}) if isinstance(payload, Mapping) else {}
        by_study = group.get("by_participant_study", {}) if isinstance(group, Mapping) else {}
        # Top-level baselines on this file
        rand = payload.get("random_baseline_expected_accuracy")
        most = (
            payload.get("baseline_most_frequent_gold_index", {}).get("accuracy")
            if isinstance(payload.get("baseline_most_frequent_gold_index"), Mapping)
            else None
        )
        for key, stats in (by_study.items() if isinstance(by_study, Mapping) else []):
            label = label_map.get(str(key))
            if not label:
                continue
            try:
                elig = int(stats.get("n_eligible", 0) or 0)
            except (TypeError, ValueError):
                elig = 0
            if elig:
                eligible_map[label] = elig
            try:
                if rand is not None and label not in random_map:
                    random_map[label] = float(rand)
            except (TypeError, ValueError):
                pass
            try:
                if most is not None and label not in mostfreq_map:
                    mostfreq_map[label] = float(most)
            except (TypeError, ValueError):
                pass

    # Prefer RLHF issue-specific metrics, then fall back to GPT-4o
    for family in ("grpo", "grail", "gpt-4o"):
        model_root = repo_root / "models" / family
        if not model_root.exists():
            continue
        candidates = _collect_next_video_candidates(model_root)
        ordered: list[Path] = sorted(
            candidates,
            key=lambda p: (0 if not _is_checkpoint_50(p) else 1, p.as_posix()),
        )
        for metrics_path in ordered:
            _apply_from_metrics(metrics_path)

    return random_map, mostfreq_map, eligible_map


# ------------------------
# Opinion readers
# ------------------------

def _read_knn_opinion_from_next(repo_root: Path) -> StudyScoresPair:
    path = repo_root / "reports" / "knn" / "opinion_from_next" / "opinion_metrics.csv"
    best_acc: dict[StudyLabel, float] = {}
    best_mae: dict[StudyLabel, float] = {}
    if not path.exists():
        return best_acc, best_mae
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study = str(row.get("study") or "").strip()
            try:
                direction = float(row.get("accuracy_after") or "")
            except (TypeError, ValueError):
                direction = None  # type: ignore[assignment]
            try:
                mae = float(row.get("mae_after") or "")
            except (TypeError, ValueError):
                mae = None  # type: ignore[assignment]
            # maximise directional accuracy; for ties on acc, prefer lower MAE
            prev_acc = best_acc.get(study)
            prev_mae = best_mae.get(study)
            if direction is None and mae is None:
                continue

            should_update = False
            if prev_acc is None:
                should_update = True
            elif direction is not None and direction > prev_acc:
                should_update = True
            elif (
                direction == prev_acc
                and mae is not None
                and (prev_mae is None or mae < prev_mae)
            ):
                should_update = True

            if should_update:
                if direction is not None:
                    best_acc[study] = direction
                if mae is not None:
                    best_mae[study] = mae
    return best_acc, best_mae


def _read_xgb_opinion_from_next(repo_root: Path) -> StudyScoresPair:
    path = repo_root / "reports" / "xgb" / "opinion_from_next" / "opinion_metrics.csv"
    acc: dict[StudyLabel, float] = {}
    mae: dict[StudyLabel, float] = {}
    if not path.exists():
        return acc, mae
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            study = str(row.get("study") or "").strip()
            try:
                acc_val = float(row.get("accuracy_after") or row.get("direction_accuracy") or "")
            except (TypeError, ValueError):
                acc_val = None  # type: ignore[assignment]
            try:
                mae_val = float(row.get("mae_after") or "")
            except (TypeError, ValueError):
                mae_val = None  # type: ignore[assignment]
            if acc_val is not None:
                acc[study] = acc_val
            if mae_val is not None:
                mae[study] = mae_val
    return acc, mae


def _read_csv_opinion(repo_root: Path, family: str) -> StudyScoresPair:
    """Read opinion CSV written by GPT-4o reports."""
    path = repo_root / "reports" / family / "opinion" / "opinion_metrics.csv"
    acc: dict[StudyLabel, float] = {}
    mae: dict[StudyLabel, float] = {}
    if not path.exists():
        return acc, mae
    label_map = _label_by_key()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_study = str(row.get("study") or "").strip()
            study = label_map.get(raw_study, raw_study)
            try:
                acc_val = float(row.get("direction_accuracy") or row.get("accuracy_after") or "")
            except (TypeError, ValueError):
                acc_val = None  # type: ignore[assignment]
            try:
                mae_val = float(row.get("mae_after") or "")
            except (TypeError, ValueError):
                mae_val = None  # type: ignore[assignment]
            if acc_val is not None:
                acc[study] = acc_val
            if mae_val is not None:
                mae[study] = mae_val
    return acc, mae


def _read_rlhf_opinion(repo_root: Path, family: str) -> StudyScoresPair:
    """Read per-study opinion metrics from RLHF artefacts.

    RLHF runs are nested under ``models/<family>/<scenario>/checkpoint-*/opinion``
    with per-study folders (``study1``, ``study2``, ...). We recursively scan the
    family directory and extract values from any matching ``metrics.json`` files,
    preferring checkpoint-50 artefacts when multiple candidates exist.
    """

    model_root = repo_root / "models" / family
    label_map = _label_by_key()
    acc: StudyScores = {}
    mae: StudyScores = {}
    if not model_root.exists():
        return acc, mae

    # Collect all metrics.json files that belong to opinion runs, excluding sweeps
    candidates: list[Path] = []
    for path in sorted(model_root.rglob("metrics.json")):
        posix = path.as_posix()
        if "/opinion/" not in posix:
            continue
        if "/sweeps/" in posix:
            continue
        candidates.append(path)

    # Order so that non-checkpoint-50 are seen first, allowing checkpoint-50 to override
    ordered: list[Path] = sorted(
        candidates,
        key=lambda p: (0 if not _is_checkpoint_50(p) else 1, p.as_posix()),
    )

    allowed_scenario = {"study1": "gun", "study2": "wage", "study3": "wage"}

    for metrics_path in ordered:
        # Study key is encoded in the directory name (e.g., .../study1/metrics.json)
        study_key: Optional[str] = None
        for part in reversed(metrics_path.parts):
            if part.startswith("study"):
                study_key = part
                break
        if not study_key:
            continue
        # Heuristic: restrict studies to their canonical issue scenario
        posix = metrics_path.as_posix()
        scenario = "gun" if "/gun/" in posix else ("wage" if "/wage/" in posix else None)
        expected = allowed_scenario.get(study_key)
        if expected is not None and scenario is not None and scenario != expected:
            continue
        label = label_map.get(study_key)
        if not label:
            continue

        payload = _load_json(metrics_path)
        metrics = payload.get("metrics", payload)
        if not isinstance(metrics, Mapping):
            continue
        # Skip entries with no eligible rows
        try:
            if int(metrics.get("eligible", 0) or 0) <= 0:
                continue
        except (TypeError, ValueError):
            pass
        try:
            direction = float(metrics.get("direction_accuracy"))
        except (TypeError, ValueError):
            direction = None  # type: ignore[assignment]
        try:
            mae_val = float(metrics.get("mae_after"))
        except (TypeError, ValueError):
            mae_val = None  # type: ignore[assignment]
        if direction is not None:
            acc[label] = direction
        if mae_val is not None:
            mae[label] = mae_val

    return acc, mae


def _gather_opinion_baselines(
    repo_root: Path,
) -> tuple[dict[StudyLabel, float], dict[StudyLabel, float], dict[StudyLabel, float], dict[StudyLabel, int]]:
    """Collect opinion baselines and eligible counts per study.

    Returns:
      - no_change_dir: direction accuracy when predicting no change (use-before baseline)
      - mae_global_mean: MAE when predicting the global mean of the after scores
      - mae_using_before: MAE when predicting the pre-study index
      - eligible_map: evaluation counts per study
    """
    label_map = _label_by_key()
    no_change_dir: dict[StudyLabel, float] = {}
    mae_global_mean: dict[StudyLabel, float] = {}
    mae_using_before: dict[StudyLabel, float] = {}
    eligible_map: dict[StudyLabel, int] = {}

    # Prefer RLHF metrics which include detailed baselines
    for family in ("grpo", "grail"):
        root = repo_root / "models" / family
        if not root.exists():
            continue
        for metrics_path in sorted(root.rglob("opinion/**/study*/metrics.json")):
            payload = _load_json(metrics_path)
            metrics = payload.get("metrics", payload)
            baseline = payload.get("baseline", {})
            if not isinstance(metrics, Mapping):
                continue
            study_key = metrics_path.parent.name
            label = label_map.get(study_key)
            if not label:
                continue
            try:
                elig = int(metrics.get("eligible", 0) or 0)
            except (TypeError, ValueError):
                elig = 0
            if elig:
                eligible_map[label] = elig
            try:
                val = float(baseline.get("direction_accuracy")) if isinstance(baseline, Mapping) else None
                if val is not None:
                    no_change_dir[label] = val
            except (TypeError, ValueError):
                pass
            try:
                gm = float(baseline.get("mae_global_mean_after")) if isinstance(baseline, Mapping) else None
                if gm is not None:
                    mae_global_mean[label] = gm
            except (TypeError, ValueError):
                pass
            try:
                ub = float(baseline.get("mae_using_before")) if isinstance(baseline, Mapping) else None
                if ub is not None:
                    mae_using_before[label] = ub
            except (TypeError, ValueError):
                pass

    # Fallback to GPT-4o CSV for no-change baseline and eligible counts
    gpt_csv = repo_root / "reports" / "gpt4o" / "opinion" / "opinion_metrics.csv"
    if gpt_csv.exists():
        import csv as _csv
        with gpt_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = _csv.DictReader(handle)
            for row in reader:
                key = str(row.get("study") or "").strip()
                label = label_map.get(key, key)
                try:
                    elig = int(row.get("eligible") or 0)
                except (TypeError, ValueError):
                    elig = 0
                if elig and label not in eligible_map:
                    eligible_map[label] = elig
                try:
                    base = float(row.get("baseline_direction_accuracy") or "")
                except (TypeError, ValueError):
                    base = None  # type: ignore[assignment]
                if base is not None and label not in no_change_dir:
                    no_change_dir[label] = base

    return no_change_dir, mae_global_mean, mae_using_before, eligible_map


# ------------------------
# Report assembly
# ------------------------

def _gather_next_video(repo_root: Path) -> dict[str, StudyScores]:
    """Gather per-family next-video accuracy maps."""
    return {
        "gpt4o": _read_grouped_json_next_video(repo_root, "gpt-4o"),
        "grpo": _read_grouped_json_next_video(repo_root, "grpo"),
        "grail": _read_grouped_json_next_video(repo_root, "grail"),
        "knn": _read_knn_next_video(repo_root),
        "xgb": _read_xgb_next_video(repo_root),
    }


def _gather_opinion(repo_root: Path) -> tuple[dict[str, StudyScores], dict[str, StudyScores]]:
    """Gather per-family opinion direction and MAE maps."""
    knn_dir, knn_mae = _read_knn_opinion_from_next(repo_root)
    xgb_dir, xgb_mae = _read_xgb_opinion_from_next(repo_root)
    gpt4o_dir, gpt4o_mae = _read_csv_opinion(repo_root, "gpt4o")
    grpo_dir, grpo_mae = _read_rlhf_opinion(repo_root, "grpo")
    grail_dir, grail_mae = _read_rlhf_opinion(repo_root, "grail")
    dir_map = {
        "gpt4o": gpt4o_dir,
        "grpo": grpo_dir,
        "grail": grail_dir,
        "knn": knn_dir,
        "xgb": xgb_dir,
    }
    mae_map = {
        "gpt4o": gpt4o_mae,
        "grpo": grpo_mae,
        "grail": grail_mae,
        "knn": knn_mae,
        "xgb": xgb_mae,
    }
    return dir_map, mae_map


def _build_opinion_dir_table(
    *,
    studies: Sequence[StudyLabel],
    n_map: Mapping[StudyLabel, int],
    no_change: Mapping[StudyLabel, float],
    gpt4o: Mapping[StudyLabel, float],
    grpo: Mapping[StudyLabel, float],
    grail: Mapping[StudyLabel, float],
    knn: Mapping[StudyLabel, float],
    xgb: Mapping[StudyLabel, float],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for study in studies:
        n = n_map.get(study)
        rand_str = _fmt_rate_ci(1 / 3, n)
        base_str = _fmt_rate_ci(no_change.get(study), n)
        p_base = no_change.get(study)
        base_bounds = None
        if p_base is not None and n:
            base_bounds = _wilson_ci(float(p_base), int(n))

        def model_cell(p: Optional[float]) -> str:
            s = _fmt_rate_ci(p, n)
            if p is None or base_bounds is None or n is None or n <= 0:
                return s
            lo_m, _ = _wilson_ci(float(p), int(n))
            _, hi_b = base_bounds
            return f"**{s}**" if lo_m > hi_b else s

        rows.append(
            [
                study,
                _fmt_int(n),
                rand_str,
                base_str,
                model_cell(gpt4o.get(study)),
                model_cell(grpo.get(study)),
                model_cell(grail.get(study)),
                model_cell(knn.get(study)),
                model_cell(xgb.get(study)),
            ]
        )
    return rows

def _build_next_video_table(
    *,
    studies: Sequence[StudyLabel],
    n_map: Mapping[StudyLabel, int],
    random_base: Mapping[StudyLabel, float],
    mostfreq_base: Mapping[StudyLabel, float],
    gpt4o: Mapping[StudyLabel, float],
    grpo: Mapping[StudyLabel, float],
    grail: Mapping[StudyLabel, float],
    knn: Mapping[StudyLabel, float],
    xgb: Mapping[StudyLabel, float],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for study in studies:
        n = n_map.get(study)
        # Baselines with CI
        rand_str = _fmt_rate_ci(random_base.get(study), n)
        most_str = _fmt_rate_ci(mostfreq_base.get(study), n)
        # Baseline bounds for significance check
        p_base = mostfreq_base.get(study)
        base_bounds = None
        if p_base is not None and n:
            base_bounds = _wilson_ci(float(p_base), int(n))

        def model_cell(p: Optional[float]) -> str:
            s = _fmt_rate_ci(p, n)
            if p is None or base_bounds is None or n is None or n <= 0:
                return s
            lo_m, _ = _wilson_ci(float(p), int(n))
            _, hi_b = base_bounds
            return f"**{s}**" if lo_m > hi_b else s

        rows.append(
            [
                study,
                _fmt_int(n),
                rand_str,
                most_str,
                model_cell(gpt4o.get(study)),
                model_cell(grpo.get(study)),
                model_cell(grail.get(study)),
                model_cell(knn.get(study)),
                model_cell(xgb.get(study)),
            ]
        )
    return rows


def _build_opinion_table(
    *,
    studies: Sequence[StudyLabel],
    n_map: Mapping[StudyLabel, int],
    gpt4o: Mapping[StudyLabel, float],
    grpo: Mapping[StudyLabel, float],
    grail: Mapping[StudyLabel, float],
    knn: Mapping[StudyLabel, float],
    xgb: Mapping[StudyLabel, float],
) -> list[list[str]]:
    rows: list[list[str]] = []
    for study in studies:
        rows.append(
            [
                study,
                _fmt_int(n_map.get(study)),
                _fmt_rate(gpt4o.get(study)),
                _fmt_rate(grpo.get(study)),
                _fmt_rate(grail.get(study)),
                _fmt_rate(knn.get(study)),
                _fmt_rate(xgb.get(study)),
            ]
        )
    return rows


def generate_portfolio_report(repo_root: Path) -> None:
    """Materialise the cross-baseline comparison report under reports/main.

    :param repo_root: Repository root used to locate metrics artefacts.
    :returns: ``None``. The Markdown report is written to disk.
    """

    studies = _study_order()

    # Next-video metrics
    next_video = _gather_next_video(repo_root)
    nv_rand, nv_most, nv_n = _gather_next_video_baselines(repo_root)

    # Opinion metrics (opinion_from_next for knn/xgb)
    op_dir, op_mae = _gather_opinion(repo_root)
    op_no_change, op_mae_mean, op_mae_using_before, op_n = _gather_opinion_baselines(repo_root)

    # Assemble Markdown
    reports_dir = repo_root / "reports" / "main"
    path, lines = start_markdown_report(reports_dir, title="Portfolio Comparison (Main)")
    lines.append(
        "This report compares grail, grpo, gpt4o, knn, and xgb across the"
        " next-video and opinion tasks. KNN and XGB opinion metrics come"
        " from their `opinion_from_next` runs."
    )
    lines.append("")

    # Next-video table
    append_markdown_table(
        lines,
        title="## Next-Video Eligible Accuracy (↑)",
        headers=["Study", "N", "Random", "Most-Freq.", "GPT-4o", "GRPO", "GRAIL", "KNN", "XGB"],
        rows=_build_next_video_table(
            studies=studies,
            n_map=nv_n,
            random_base=nv_rand,
            mostfreq_base=nv_most,
            gpt4o=next_video["gpt4o"],
            grpo=next_video["grpo"],
            grail=next_video["grail"],
            knn=next_video["knn"],
            xgb=next_video["xgb"],
        ),
        empty_message="No next-video metrics available.",
    )

    # Opinion tables
    append_markdown_table(
        lines,
        title="## Opinion Directional Accuracy (↑)",
        headers=["Study", "N", "Random (1/3)", "No-change", "GPT-4o", "GRPO", "GRAIL", "KNN", "XGB"],
        rows=_build_opinion_dir_table(
            studies=studies,
            n_map=op_n,
            no_change=op_no_change,
            gpt4o=op_dir["gpt4o"],
            grpo=op_dir["grpo"],
            grail=op_dir["grail"],
            knn=op_dir["knn"],
            xgb=op_dir["xgb"],
        ),
        empty_message="No opinion directional-accuracy metrics available.",
    )

    append_markdown_table(
        lines,
        title="## Opinion MAE (↓)",
        headers=["Study", "N", "Global Mean", "Using Before", "GPT-4o", "GRPO", "GRAIL", "KNN", "XGB"],
        rows=[
            [
                row[0],
                row[1],
                _fmt_rate(op_mae_mean.get(row[0])),
                _fmt_rate(op_mae_using_before.get(row[0])),
                *row[2:],
            ]
            for row in _build_opinion_table(
                studies=studies,
                n_map=op_n,
                gpt4o=op_mae["gpt4o"],
                grpo=op_mae["grpo"],
                grail=op_mae["grail"],
                knn=op_mae["knn"],
                xgb=op_mae["xgb"],
            )
        ],
        empty_message="No opinion MAE metrics available.",
    )

    lines.extend(
        [
            "Notes",
            "",
            (
                "- KNN/XGB opinion metrics reflect training on next-video "
                "representations (`opinion_from_next`)."
            ),
            (
                "- GPT-4o next-video accuracies and per-study opinion CSVs "
                "are sourced from their report artefacts."
            ),
            (
                "- GRPO/GRAIL metrics are read from `models/<family>/` caches "
                "when available."
            ),
            (
                "- Baselines: For next-video, 'Random' is the expected accuracy "
                "of uniformly picking among the slate; 'Most-Freq.' always "
                "chooses the most common gold index in the split. For opinion "
                "direction, 'Random (1/3)' assumes equal probability of up/none/down; "
                "'No-change' predicts the pre-study opinion. For opinion MAE, "
                "'Global Mean' predicts the dataset mean of the post-study index; "
                "'Using Before' predicts the pre-study index (when available)."
            ),
            (
                "- Column 'N' reports the number of eligible evaluation examples per study."
            ),
            (
                "- Accuracy entries include 95% Wilson CIs in brackets. Bold "
                "indicates the model's lower CI exceeds the baseline's upper "
                "CI (Most‑Freq. for next‑video; No‑change for opinion)."
            ),
            "",
        ]
    )

    write_markdown_lines(path, lines)


def main(_argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI convenience
    """CLI entry point that regenerates the portfolio report.

    :param _argv: Optional sequence of CLI tokens (unused).
    :returns: ``None``.
    """
    repo_root = Path(__file__).resolve().parents[3]
    generate_portfolio_report(repo_root)


__all__ = ["generate_portfolio_report", "main"]

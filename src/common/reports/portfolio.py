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
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence

from .utils import start_markdown_report
from .tables import append_markdown_table
from common.pipeline.io import write_markdown_lines
from common.opinion import DEFAULT_SPECS


ModelKey = str  # One of: "gpt4o", "grpo", "grail", "knn", "xgb"
StudyLabel = str  # e.g. "Study 1 – Gun Control (MTurk)"


@dataclass(frozen=True)
class MetricBundle:
    next_video_accuracy: Optional[float] = None
    opinion_direction: Optional[float] = None
    opinion_mae: Optional[float] = None


def _fmt_rate(value: Optional[float], digits: int = 3) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _study_order() -> list[StudyLabel]:
    return [spec.label for spec in DEFAULT_SPECS]


def _label_by_key() -> dict[str, str]:
    return {spec.key: spec.label for spec in DEFAULT_SPECS}


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
    """Return per-study accuracy from RLHF/GPT-4o JSON metrics."""
    root = repo_root / "models" / family / "next_video"
    label_map = _label_by_key()
    values: dict[StudyLabel, float] = {}
    if not root.exists():
        return values
    # Prefer direct next_video runs over sweeps
    candidates: list[Path] = []
    for sub in sorted(root.glob("*/metrics.json")):
        if "/sweeps/" in sub.as_posix():
            continue
        candidates.append(sub)
    if not candidates:
        # fall back to any metrics under next_video
        candidates = sorted(root.rglob("metrics.json"))
    if not candidates:
        return values
    metrics = _load_json(candidates[0])
    group = (
        metrics.get("group_metrics", {})
        if isinstance(metrics, Mapping)
        else {}
    )
    by_study = (
        group.get("by_participant_study", {})
        if isinstance(group, Mapping)
        else {}
    )
    if not isinstance(by_study, Mapping):
        return values
    for key, payload in by_study.items():
        if not isinstance(payload, Mapping):
            continue
        label = label_map.get(str(key))
        if not label:
            continue
        try:
            acc = float(payload.get("accuracy"))
        except (TypeError, ValueError):
            continue
        values[label] = acc
    return values


# ------------------------
# Opinion readers
# ------------------------

def _read_knn_opinion_from_next(repo_root: Path) -> tuple[dict[StudyLabel, float], dict[StudyLabel, float]]:
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
            if prev_acc is None or (direction is not None and direction > prev_acc) or (
                direction == prev_acc and mae is not None and (prev_mae is None or mae < prev_mae)
            ):
                if direction is not None:
                    best_acc[study] = direction
                if mae is not None:
                    best_mae[study] = mae
    return best_acc, best_mae


def _read_xgb_opinion_from_next(repo_root: Path) -> tuple[dict[StudyLabel, float], dict[StudyLabel, float]]:
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


def _read_csv_opinion(repo_root: Path, family: str) -> tuple[dict[StudyLabel, float], dict[StudyLabel, float]]:
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


def _read_rlhf_opinion(repo_root: Path, family: str) -> tuple[dict[StudyLabel, float], dict[StudyLabel, float]]:
    """Read per-study opinion metrics from RLHF artefacts."""
    root = repo_root / "models" / family / "opinion"
    label_map = _label_by_key()
    acc: dict[StudyLabel, float] = {}
    mae: dict[StudyLabel, float] = {}
    if not root.exists():
        return acc, mae
    # search for the first run label that contains study subdirs
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    for run in run_dirs:
        study_dirs = sorted([p for p in run.iterdir() if p.is_dir()])
        if not study_dirs:
            continue
        for study_dir in study_dirs:
            metrics_path = study_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            payload = _load_json(metrics_path)
            metrics = payload.get("metrics", payload)
            if not isinstance(metrics, Mapping):
                continue
            key = study_dir.name
            label = label_map.get(key)
            if not label:
                continue
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
        # Only consider the first populated run label
        if acc or mae:
            break
    return acc, mae


# ------------------------
# Report assembly
# ------------------------

def _build_next_video_table(
    *,
    studies: Sequence[StudyLabel],
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
                _fmt_rate(gpt4o.get(study)),
                _fmt_rate(grpo.get(study)),
                _fmt_rate(grail.get(study)),
                _fmt_rate(knn.get(study)),
                _fmt_rate(xgb.get(study)),
            ]
        )
    return rows


def _build_opinion_table(
    *,
    studies: Sequence[StudyLabel],
    metric_key: str,
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
    knn_next = _read_knn_next_video(repo_root)
    xgb_next = _read_xgb_next_video(repo_root)
    gpt4o_next = _read_grouped_json_next_video(repo_root, "gpt-4o")
    grpo_next = _read_grouped_json_next_video(repo_root, "grpo")
    grail_next = _read_grouped_json_next_video(repo_root, "grail")

    # Opinion metrics (opinion_from_next for knn/xgb)
    knn_op_dir, knn_op_mae = _read_knn_opinion_from_next(repo_root)
    xgb_op_dir, xgb_op_mae = _read_xgb_opinion_from_next(repo_root)
    gpt4o_op_dir, gpt4o_op_mae = _read_csv_opinion(repo_root, "gpt4o")
    grpo_op_dir, grpo_op_mae = _read_rlhf_opinion(repo_root, "grpo")
    grail_op_dir, grail_op_mae = _read_rlhf_opinion(repo_root, "grail")

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
        headers=["Study", "GPT-4o", "GRPO", "GRAIL", "KNN", "XGB"],
        rows=_build_next_video_table(
            studies=studies,
            gpt4o=gpt4o_next,
            grpo=grpo_next,
            grail=grail_next,
            knn=knn_next,
            xgb=xgb_next,
        ),
        empty_message="No next-video metrics available.",
    )

    # Opinion tables
    append_markdown_table(
        lines,
        title="## Opinion Directional Accuracy (↑)",
        headers=["Study", "GPT-4o", "GRPO", "GRAIL", "KNN", "XGB"],
        rows=_build_opinion_table(
            studies=studies,
            metric_key="direction_accuracy",
            gpt4o=gpt4o_op_dir,
            grpo=grpo_op_dir,
            grail=grail_op_dir,
            knn=knn_op_dir,
            xgb=xgb_op_dir,
        ),
        empty_message="No opinion directional-accuracy metrics available.",
    )

    append_markdown_table(
        lines,
        title="## Opinion MAE (↓)",
        headers=["Study", "GPT-4o", "GRPO", "GRAIL", "KNN", "XGB"],
        rows=_build_opinion_table(
            studies=studies,
            metric_key="mae_after",
            gpt4o=gpt4o_op_mae,
            grpo=grpo_op_mae,
            grail=grail_op_mae,
            knn=knn_op_mae,
            xgb=xgb_op_mae,
        ),
        empty_message="No opinion MAE metrics available.",
    )

    lines.extend(
        [
            "Notes",
            "",
            "- KNN/XGB opinion metrics reflect training on next-video representations (`opinion_from_next`).",
            "- GPT-4o next-video accuracies and per-study opinion CSVs are sourced from their report artefacts.",
            "- GRPO/GRAIL metrics are read from `models/<family>/` caches when available.",
            "",
        ]
    )

    write_markdown_lines(path, lines)


def main(argv: Sequence[str] | None = None) -> None:  # pragma: no cover - CLI convenience
    repo_root = Path(__file__).resolve().parents[3]
    generate_portfolio_report(repo_root)


__all__ = ["generate_portfolio_report", "main"]

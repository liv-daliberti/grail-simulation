#!/usr/bin/env python
"""Aggregate RLHF (GRPO/GRAIL) reports across issue-specific runs.

This utility reads existing next-video and opinion artifacts from
`models/<family>/{gun,wage}/checkpoint-*/**` and materialises a single report
for the family under `reports/<family>/`—mirroring the GPT-4o layout.

Usage (from repo root):
  python -m common.rlhf.aggregate_family_report --family grpo
  python -m common.rlhf.aggregate_family_report --family grail
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Mapping, Optional

from .reports import ReportOptions, generate_reports as _generate_reports


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> Mapping[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _collect_next_video_metrics(family: str) -> dict[str, tuple[Path, Mapping[str, object]]]:
    """Return per-issue metrics.json payloads for ``family``.

    Chooses the most recently modified file when multiple candidates exist.
    """

    model_root = _repo_root() / "models" / family
    issue_payloads: dict[str, tuple[Path, Mapping[str, object]]] = {}
    for metrics_path in sorted(model_root.rglob("next_video/**/metrics.json")):
        # Identify issue from parent dataset or metrics content
        payload = _load_json(metrics_path)
        metrics = payload.get("metrics", payload)
        dataset = str(metrics.get("dataset") or "")
        issue = ""
        if "minimum_wage" in dataset:
            issue = "minimum_wage"
        elif "gun_control" in dataset:
            issue = "gun_control"
        # Fallback to group key if dataset hint is missing
        if not issue:
            by_issue = metrics.get("group_metrics", {}).get("by_issue", {})
            if isinstance(by_issue, Mapping) and by_issue:
                issue = next(iter(by_issue.keys()))
        if not issue:
            continue
        # Prefer the latest metrics for each issue
        prev = issue_payloads.get(issue)
        if prev is None or metrics_path.stat().st_mtime > prev[0].stat().st_mtime:
            issue_payloads[issue] = (metrics_path, metrics)
    return issue_payloads


def _merge_next_video_metrics(payloads: dict[str, Mapping[str, object]]) -> Mapping[str, object]:
    """Combine per-issue next-video metrics into a single mapping."""

    n_total = 0
    n_eligible = 0
    parsed_weight = 0.0
    format_weight = 0.0
    total_correct = 0

    by_issue: dict[str, dict[str, object]] = {}
    # For studies, accumulate seen/eligible and compute weighted metrics.
    accum_by_study: dict[str, dict[str, float]] = {}

    for issue, metrics in payloads.items():
        # Unwrap nested "metrics" when present
        m = metrics
        try:
            n_total += int(m.get("n_total", 0))
            n_eligible += int(m.get("n_eligible", 0))
            parsed = float(m.get("parsed_rate", 0.0) or 0.0)
            fmt = float(m.get("format_rate", 0.0) or 0.0)
            parsed_weight += parsed * int(m.get("n_total", 0))
            format_weight += fmt * int(m.get("n_total", 0))
        except (TypeError, ValueError):
            pass

        # Track correct counts from by_issue if available
        group = m.get("group_metrics", {}) if isinstance(m, Mapping) else {}
        g_issue = group.get("by_issue", {}) if isinstance(group, Mapping) else {}
        if isinstance(g_issue, Mapping) and issue in g_issue:
            try:
                total_correct += int(g_issue[issue].get("correct", 0))
            except (TypeError, ValueError):
                pass
            by_issue[issue] = dict(g_issue[issue])

        g_study = group.get("by_participant_study", {}) if isinstance(group, Mapping) else {}
        if isinstance(g_study, Mapping):
            for key, val in g_study.items():
                try:
                    n_seen = int(val.get("n_seen", 0) or 0)
                    n_elig = int(val.get("n_eligible", 0) or 0)
                    acc = float(val.get("accuracy", 0.0) or 0.0)
                    parsed = float(val.get("parsed_rate", 0.0) or 0.0)
                    fmt = float(val.get("format_rate", 0.0) or 0.0)
                except (TypeError, ValueError):
                    continue
                acc_item = accum_by_study.setdefault(
                    key,
                    {"n_seen": 0.0, "n_eligible": 0.0, "acc_w": 0.0, "parsed_w": 0.0, "format_w": 0.0},
                )
                acc_item["n_seen"] += float(n_seen)
                acc_item["n_eligible"] += float(n_elig)
                acc_item["acc_w"] += acc * float(n_elig)
                acc_item["parsed_w"] += parsed * float(n_seen)
                acc_item["format_w"] += fmt * float(n_seen)

    # Fallback: if per-issue 'correct' wasn't available, approximate from accuracy_overall
    if total_correct == 0:
        for m in payloads.values():
            try:
                acc_i = float(m.get("accuracy_overall", 0.0) or 0.0)
                elig_i = int(m.get("n_eligible", 0) or 0)
                total_correct += int(round(acc_i * elig_i))
            except (TypeError, ValueError):
                continue

    accuracy_overall = (total_correct / n_eligible) if n_eligible > 0 and total_correct else None
    parsed_rate = (parsed_weight / n_total) if n_total > 0 else None
    format_rate = (format_weight / n_total) if n_total > 0 else None

    # Finalise by-study weighted rows
    by_study: dict[str, dict[str, object]] = {}
    for key, acc_item in accum_by_study.items():
        n_seen = int(acc_item.get("n_seen", 0.0))
        n_elig = int(acc_item.get("n_eligible", 0.0))
        acc = (acc_item.get("acc_w", 0.0) / n_elig) if n_elig else None
        parsed = (acc_item.get("parsed_w", 0.0) / n_seen) if n_seen else None
        fmt = (acc_item.get("format_w", 0.0) / n_seen) if n_seen else None
        by_study[key] = {
            "n_seen": n_seen,
            "n_eligible": n_elig,
            "accuracy": acc,
            "parsed_rate": parsed,
            "format_rate": fmt,
        }

    return {
        "n_total": n_total,
        "n_eligible": n_eligible,
        "accuracy_overall": accuracy_overall,
        "parsed_rate": parsed_rate,
        "format_rate": format_rate,
        "group_metrics": {
            "by_issue": by_issue,
            "by_participant_study": by_study,
        },
        "notes": "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.",
    }


@dataclass(frozen=True)
class _NVResult:
    run_dir: Path
    metrics_path: Path
    predictions_path: Path
    qa_log_path: Path
    metrics: Mapping[str, object]


def _build_next_video_result(family: str) -> Optional[_NVResult]:
    issue_map = _collect_next_video_metrics(family)
    if not issue_map:
        return None
    combined_metrics = _merge_next_video_metrics({k: v for k, (_, v) in issue_map.items()})

    # Pick an arbitrary predictions/qa path to enable sample gallery links
    any_path = next(iter(issue_map.values()))[0]
    run_dir = any_path.parent
    preds = run_dir / "predictions.jsonl"
    qa_log = _repo_root() / "logs" / family / "qa.log"  # placeholder; not used in report body
    return _NVResult(
        run_dir=run_dir,
        metrics_path=any_path,
        predictions_path=preds if preds.exists() else run_dir / "predictions.jsonl",
        qa_log_path=qa_log,
        metrics=combined_metrics,
    )


def _collect_opinion_studies(family: str) -> list[dict[str, object]]:
    """Return per-study opinion payloads from both gun and wage runs."""

    model_root = _repo_root() / "models" / family
    studies: dict[str, dict[str, object]] = {}
    for metrics_path in sorted(model_root.rglob("opinion/**/study*/metrics.json")):
        payload = _load_json(metrics_path)
        metrics = payload.get("metrics", payload)
        baseline = payload.get("baseline", {})
        study_key = metrics_path.parent.name
        predictions = metrics_path.parent / "predictions.jsonl"
        row = {
            "study_key": study_key,
            "metrics": metrics,
            "baseline": baseline if isinstance(baseline, Mapping) else {},
            "participants": int(metrics.get("participants", 0) or 0),
            "eligible": int(metrics.get("eligible", 0) or 0),
            "predictions": predictions,
        }
        # Prefer latest file for a given study_key
        prev = studies.get(study_key)
        if prev is None or metrics_path.stat().st_mtime > (prev.get("_mtime") or 0):
            row["_mtime"] = metrics_path.stat().st_mtime
            studies[study_key] = row
    # Drop helper field
    for v in studies.values():
        v.pop("_mtime", None)
    return list(studies.values())


def _combine_opinion_metrics(rows: list[dict[str, object]]) -> Mapping[str, object]:
    """Compute combined opinion metrics across studies using eligible-weighting."""

    total_elig = sum(int(r.get("eligible", 0) or 0) for r in rows)
    if total_elig <= 0:
        return {}

    def _get_float(r: dict, key: str) -> float:
        try:
            return float(r.get("metrics", {}).get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    # Weighted sums
    w_mae_after = sum(_get_float(r, "mae_after") * int(r.get("eligible", 0) or 0) for r in rows)
    w_mae_change = sum(_get_float(r, "mae_change") * int(r.get("eligible", 0) or 0) for r in rows)
    w_dir_ok = sum(_get_float(r, "direction_accuracy") * int(r.get("eligible", 0) or 0) for r in rows)

    # RMSE requires SSE aggregation: rmse^2 * n
    sse_after = sum(((_get_float(r, "rmse_after") ** 2) * int(r.get("eligible", 0) or 0)) for r in rows)
    sse_change = sum(((_get_float(r, "rmse_change") ** 2) * int(r.get("eligible", 0) or 0)) for r in rows)

    # Calibration ECE: approximate with eligible-weighted average
    w_ece = sum(_get_float(r, "calibration_ece") * int(r.get("eligible", 0) or 0) for r in rows)

    return {
        "eligible": total_elig,
        "mae_after": (w_mae_after / total_elig) if total_elig else None,
        "mae_change": (w_mae_change / total_elig) if total_elig else None,
        "direction_accuracy": (w_dir_ok / total_elig) if total_elig else None,
        "rmse_after": sqrt(sse_after / total_elig) if total_elig and sse_after else None,
        "rmse_change": sqrt(sse_change / total_elig) if total_elig and sse_change else None,
        "calibration_ece": (w_ece / total_elig) if total_elig else None,
    }


@dataclass(frozen=True)
class _OpinionStudy:
    study_label: str
    participants: int
    eligible: int
    metrics: Mapping[str, object]
    baseline: Mapping[str, object]
    artifacts: object  # expects .predictions Path for sample gallery


@dataclass(frozen=True)
class _OpinionArtifacts:
    predictions: Path
    qa_log: Path


@dataclass(frozen=True)
class _OpinionResult:
    studies: list[_OpinionStudy]
    combined_metrics: Mapping[str, object]


def _build_opinion_result(family: str) -> Optional[_OpinionResult]:
    rows = _collect_opinion_studies(family)
    if not rows:
        return None

    label_map = {
        "study1": "Study 1 – Gun Control (MTurk)",
        "study2": "Study 2 – Minimum Wage (MTurk)",
        "study3": "Study 3 – Minimum Wage (YouGov)",
    }
    studies: list[_OpinionStudy] = []
    for r in rows:
        key = str(r.get("study_key"))
        studies.append(
            _OpinionStudy(
                study_label=label_map.get(key, key),
                participants=int(r.get("participants", 0) or 0),
                eligible=int(r.get("eligible", 0) or 0),
                metrics=r.get("metrics", {}),
                baseline=r.get("baseline", {}),
                artifacts=_OpinionArtifacts(
                    predictions=r.get("predictions"),
                    qa_log=r.get("predictions", Path(".")).with_name("qa.log"),
                ),
            )
        )

    combined = _combine_opinion_metrics(rows)
    return _OpinionResult(studies=studies, combined_metrics=combined)


def main(argv: Optional[list[str]] = None) -> None:  # pragma: no cover - CLI convenience
    parser = argparse.ArgumentParser(description="Aggregate RLHF reports across issues")
    parser.add_argument("--family", choices=["grpo", "grail"], required=True)
    args = parser.parse_args(argv)

    family = args.family
    repo_root = _repo_root()

    nv = _build_next_video_result(family)
    op = _build_opinion_result(family)

    _generate_reports(
        repo_root=repo_root,
        next_video=nv,
        opinion=op,
        options=ReportOptions(
            reports_subdir=family,
            baseline_label=family.upper(),
            regenerate_hint=(
                "Regenerate via: "
                f"python -m common.rlhf.aggregate_family_report --family {family}"
            ),
        ),
    )


if __name__ == "__main__":  # pragma: no cover
    main()

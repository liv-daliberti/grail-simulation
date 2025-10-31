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

"""Markdown report builders shared by GRPO and GRAIL baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from common.pipeline.io import write_markdown_lines
from common.reports.samples import write_sample_responses_report

from grpo.next_video import NextVideoEvaluationResult
from grpo.opinion import OpinionEvaluationResult

DEFAULT_BASELINE_LABEL = "RLHF"
DEFAULT_REGENERATE_HINT = None


@dataclass(frozen=True)
class ReportOptions:
    """Configuration controlling how RLHF reports are rendered.

    :param reports_subdir: Subdirectory under ``reports/`` where artefacts are written.
    :param baseline_label: Human-readable label used in report headings.
    :param regenerate_hint: Optional hint describing how to regenerate artefacts.
    """

    reports_subdir: str = "rlhf"
    baseline_label: str = DEFAULT_BASELINE_LABEL
    regenerate_hint: str | None = DEFAULT_REGENERATE_HINT


def _format_rate(value: float | int | None, precision: int = 3) -> str:
    """Return a formatted rate with the requested precision.

    :param value: Numeric value representing a rate.
    :param precision: Number of decimal places to display.
    :returns: Formatted rate string or an em dash when ``value`` is falsy.
    """

    if value is None:
        return "—"
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return "—"


def _format_int(value: int | float | None) -> str:
    """Return an integer string with thousand separators.

    :param value: Integer or numeric value to format.
    :returns: Formatted integer string or an em dash when ``value`` is falsy.
    """

    if value is None:
        return "—"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _write_catalog_readme(
    reports_root: Path,
    baseline_label: str,
    regenerate_hint: str | None,
) -> None:
    """Materialise the top-level reports README.

    :param reports_root: Directory where report artefacts are stored.
    :param baseline_label: Human-readable name of the baseline.
    :param regenerate_hint: Optional message describing how to regenerate reports.
    :returns: ``None``. Markdown README is written to disk.
    """

    lines = [
        f"# {baseline_label} Report Catalog",
        "",
        f"Finetuned {baseline_label} evaluation artifacts:",
        "",
        "- `next_video/` – slate-ranking metrics for the configured checkpoint.",
        "- `opinion/` – opinion regression metrics across participant studies.",
        "- `sample_generative_responses/README.md` – curated examples showing the exact",
        "  prompts given to the model and the model's <think>/<answer> (and <opinion>)",
        "  outputs, with explanatory notes.",
    ]
    if regenerate_hint:
        lines.extend(["", regenerate_hint, ""])
    write_markdown_lines(reports_root / "README.md", lines)


def _write_next_video_report(
    reports_root: Path,
    result: NextVideoEvaluationResult,
    baseline_label: str,
) -> None:
    """Render the next-video evaluation summary.

    :param reports_root: Directory where report artefacts are stored.
    :param result: Evaluation metrics for the next-video baseline.
    :param baseline_label: Human-readable name of the baseline.
    :returns: ``None``. Markdown report is written to disk.
    """

    metrics = result.metrics
    accuracy = metrics.get("accuracy_overall")
    parsed_rate = metrics.get("parsed_rate")
    format_rate = metrics.get("format_rate")
    eligible = metrics.get("n_eligible")
    total = metrics.get("n_total")

    lines = [
        f"# {baseline_label} Next-Video Baseline",
        "",
        f"- **Overall accuracy:** {_format_rate(accuracy)} on {_format_int(eligible)} "
        f"eligible slates out of {_format_int(total)} processed.",
        f"- **Parsed rate:** {_format_rate(parsed_rate)}",
        f"- **Formatted rate:** {_format_rate(format_rate)}",
        "",
        "## Accuracy by Issue",
        "",
        "| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    by_issue = metrics.get("group_metrics", {}).get("by_issue", {})
    for group, stats in by_issue.items():
        lines.append(
            f"| {group} | {_format_int(stats.get('n_seen'))} | "
            f"{_format_int(stats.get('n_eligible'))} | "
            f"{_format_rate(stats.get('accuracy'))} | "
            f"{_format_rate(stats.get('parsed_rate'))} | "
            f"{_format_rate(stats.get('format_rate'))} |"
        )

    lines.extend(
        [
            "",
            "## Accuracy by Participant Study",
            "",
            "| Group | Seen | Eligible | Accuracy ↑ | Parsed ↑ | Formatted ↑ |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_study = metrics.get("group_metrics", {}).get("by_participant_study", {})
    # Ensure stable, human-friendly ordering for study keys (study1, study2, ...).
    def _study_sort_key(k: str) -> tuple[int, str]:
        if k.startswith("study"):
            suffix = k[5:]
            try:
                return (0, f"{int(suffix):03d}")
            except (TypeError, ValueError):
                return (1, k)
        return (1, k)
    for group in sorted(by_study.keys(), key=_study_sort_key):
        stats = by_study[group]
        lines.append(
            f"| {group} | {_format_int(stats.get('n_seen'))} | "
            f"{_format_int(stats.get('n_eligible'))} | "
            f"{_format_rate(stats.get('accuracy'))} | "
            f"{_format_rate(stats.get('parsed_rate'))} | "
            f"{_format_rate(stats.get('format_rate'))} |"
        )

    lines.extend(
        [
            "",
            "### Notes",
            "",
            "Accuracy computed over eligible rows (gold_index>0). Buckets: 1..4, 5+, unknown.",
            "",
        ]
    )
    write_markdown_lines(reports_root / "next_video" / "README.md", lines)


def _combined_metric_rows(combined: Mapping[str, float | int | None]) -> Iterable[tuple[str, str]]:
    """Return formatted combined-metric rows for the opinion report table.

    :param combined: Mapping containing aggregated opinion metrics.
    :returns: Iterable of label/value tuples rendered in the combined metrics table.
    """

    yield "Eligible", _format_int(combined.get("eligible"))
    yield "MAE (post-study)", _format_rate(combined.get("mae_after"))
    yield "MAE (change)", _format_rate(combined.get("mae_change"))
    yield "Direction accuracy", _format_rate(combined.get("direction_accuracy"))
    yield "RMSE (post-study)", _format_rate(combined.get("rmse_after"))
    yield "RMSE (change)", _format_rate(combined.get("rmse_change"))
    yield "Calibration ECE", _format_rate(combined.get("calibration_ece"))


def _study_rows(studies) -> Iterable[str]:
    """Yield formatted Markdown rows for per-study opinion metrics.

    :param studies: Mapping or iterable containing per-study results.
    :returns: Iterable of Markdown table rows.
    """

    study_iter = studies.values() if hasattr(studies, "values") else studies
    for study_result in study_iter:
        metrics = study_result.metrics
        baseline = study_result.baseline
        study_label = study_result.spec.label if study_result.spec else study_result.study_label
        eligible = study_result.eligible or metrics.get("eligible")
        yield (
            f"| {study_label} | {_format_int(study_result.participants)} | "
            f"{_format_int(eligible)} | "
            f"{_format_rate(metrics.get('mae_after'))} | "
            f"{_format_rate(baseline.get('mae_after'))} | "
            f"{_format_rate(metrics.get('direction_accuracy'))} | "
            f"{_format_rate(baseline.get('direction_accuracy'))} |"
        )


def _write_opinion_report(
    reports_root: Path,
    result: OpinionEvaluationResult,
    baseline_label: str,
) -> None:
    """Render the opinion regression summary.

    :param reports_root: Directory where report artefacts are stored.
    :param result: Evaluation metrics for the opinion baseline.
    :param baseline_label: Human-readable name of the baseline.
    :returns: ``None``. Markdown report is written to disk.
    """

    lines = [
        f"# {baseline_label} Opinion Regression",
        "",
        "Opinion-shift evaluation across the canonical participant studies. "
        "Baseline metrics treat the pre-study opinion index as the prediction.",
        "",
        "## Combined Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for label, value in _combined_metric_rows(result.combined_metrics):
        lines.append(f"| {label} | {value} |")

    lines.extend(
        [
            "",
            "## Per-Study Breakdown",
            "",
            (
                "| Study | Participants | Eligible | MAE ↓ | Baseline MAE ↓ | "
                "Direction ↑ | Baseline Direction ↑ |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )

    for row in _study_rows(result.studies):
        lines.append(row)

    lines.append("")
    write_markdown_lines(reports_root / "opinion" / "README.md", lines)


def generate_reports(
    *,
    repo_root: Path,
    next_video: NextVideoEvaluationResult | None,
    opinion: OpinionEvaluationResult | None,
    options: ReportOptions | None = None,
) -> None:
    """Materialise Markdown reports for RL fine-tuning baselines.

    :param repo_root: Root of the repository where reports are rendered.
    :param next_video: Optional next-video evaluation artefacts.
    :param opinion: Optional opinion evaluation artefacts.
    :param options: Optional configuration controlling output locations and labels.
    :returns: ``None``. Markdown reports are generated on disk.
    """

    opts = options or ReportOptions()
    reports_root = repo_root / "reports" / opts.reports_subdir
    (reports_root / "next_video").mkdir(parents=True, exist_ok=True)
    (reports_root / "opinion").mkdir(parents=True, exist_ok=True)

    _write_catalog_readme(reports_root, opts.baseline_label, opts.regenerate_hint)
    if next_video is not None:
        _write_next_video_report(reports_root, next_video, opts.baseline_label)
    if opinion is not None:
        _write_opinion_report(reports_root, opinion, opts.baseline_label)

    # Always attempt to render a small sample gallery from available artefacts.
    nv_files = [next_video.predictions_path] if next_video is not None else []
    op_files = [s.artifacts.predictions for s in (opinion.studies if opinion else [])]
    write_sample_responses_report(
        reports_root=reports_root,
        family_label=opts.baseline_label,
        next_video_files=nv_files,
        opinion_files=op_files,
        per_issue=5,
    )

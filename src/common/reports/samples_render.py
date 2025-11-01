"""Helpers to render human-readable sample galleries for reports.

This module turns structured :class:`~common.reports.samples_types.Sample` objects
into Markdown sections that show the original question and the model's exact
structured response blocks (``<think>``/``<answer>``), with concise notes.

The functions return lists of lines rather than writing files directly so that
callers can compose outputs flexibly (e.g., into a single Markdown document).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .samples_types import Sample


def build_header_lines(family_label: str) -> List[str]:
    """Build the common header for a sample gallery.

    The header introduces the gallery for a particular model family (e.g.,
    "XGBoost", "DistilGPT") and explains what each section contains.

    :param family_label: Human-friendly model family name appearing in the H1.
    :returns: Lines of Markdown ready to be joined with newlines.
    """
    lines: List[str] = [f"# {family_label} Sample Generative Model Responses", ""]
    lines.extend(
        [
            "This gallery shows concrete questions given to the model and the",
            "exact structured <think>/<answer> outputs it produced. Each example",
            "adds explicit notes clarifying what the model did (selection or",
            "opinion prediction), whether tags are present, and a short rationale",
            "summarised from the <think> block.",
            "",
            "Sections are grouped by issue and each includes up to 5 examples.",
            "",
        ]
    )
    return lines


def format_sample_block(idx: int, sample: Sample) -> List[str]:
    """Format a single sample as a Markdown block.

    The block includes the question, the raw structured model output wrapped in
    a fenced code block, and a short notes list with helpful metadata.

    :param idx: 1-based example index used in the section heading.
    :param sample: The sample record containing question and model outputs.
    :returns: Lines of Markdown representing the example.
    """
    lines: List[str] = []
    lines.append(f"### Example {idx} ({sample.task.replace('_', ' ').title()})")
    lines.append("")
    lines.append("#### Question")
    lines.append("")
    lines.append("```text")
    lines.append(sample.question)
    lines.append("```")
    lines.append("")
    lines.append("#### Model Response")
    lines.append("")
    lines.append("```text")
    if sample.think:
        lines.append("<think>")
        lines.append(sample.think)
        lines.append("</think>")
        lines.append("")
    if sample.answer:
        lines.append("<answer>")
        lines.append(sample.answer)
        lines.append("</answer>")
        lines.append("")
    if sample.task == "opinion" and sample.opinion_label:
        lines.append(f"<opinion>{sample.opinion_label}</opinion>")
    lines.append("```")
    lines.append("")

    def _bool_label(flag: bool) -> str:
        return "yes" if flag else "no"

    has_think = bool(sample.think)
    has_answer = bool(sample.answer)

    lines.append("#### Notes")
    lines.append("")
    lines.append(f"- Issue: {sample.issue.replace('_', ' ')}")
    task_label = (
        "Next-video selection" if sample.task == "next_video" else "Opinion shift prediction"
    )
    lines.append(f"- Task: {task_label}")
    lines.append(
        f"- Tags — think: {_bool_label(has_think)}, answer: {_bool_label(has_answer)}"
    )
    if sample.task == "next_video":
        chosen = (
            sample.chosen_option if sample.chosen_option is not None else sample.answer.strip()
        )
        lines.append(f"- Chosen option: {chosen}")
    else:  # opinion
        if sample.before is not None:
            lines.append(f"- Pre-study opinion index: {sample.before:.2f}")
        if sample.predicted_after is not None:
            lines.append(f"- Predicted post-study index: {sample.predicted_after:.2f}")
        if sample.opinion_label:
            lines.append(f"- Predicted direction: {sample.opinion_label}")

    if sample.think:
        snippet = sample.think.strip().splitlines()[0]
        if len(snippet) > 240:
            snippet = snippet[:237] + "..."
        lines.append(f"- Short rationale: {snippet}")

    return lines


def render_sections_by_issue(
    select_gun: Sequence[Sample], select_wage: Sequence[Sample]
) -> List[str]:
    """Render issue-grouped sections for gun-control and minimum-wage.

    The resulting lines contain up to two H2 sections (one per issue), each with
    numbered examples. Samples are ordered by task (opinion blocks first) and by
    a short question prefix for stable, readable ordering.

    :param select_gun: Selected examples for the gun-control issue.
    :param select_wage: Selected examples for the minimum-wage issue.
    :returns: Lines of Markdown for the grouped sections.
    """
    lines: List[str] = []
    sections = [
        ("gun_control", "Gun Control", list(select_gun)),
        ("minimum_wage", "Minimum Wage", list(select_wage)),
    ]
    for _issue_key, issue_label, selected in sections:
        if not selected:
            continue
        lines.append(f"## {issue_label}")
        lines.append("")

        def _order_key(sample: Sample) -> Tuple[int, str]:
            return (0 if sample.task == "opinion" else 1, sample.question[:60])

        for idx, sample in enumerate(sorted(selected, key=_order_key), start=1):
            lines.extend(format_sample_block(idx, sample))
    return lines


__all__ = [
    "build_header_lines",
    "format_sample_block",
    "render_sections_by_issue",
]

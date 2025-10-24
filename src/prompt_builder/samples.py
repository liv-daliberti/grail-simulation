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

"""Prompt sampling helpers for the Grail recommendation datasets."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

try:  # pragma: no cover - optional dependency
    from datasets import DatasetDict, load_from_disk  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    DatasetDict = None  # type: ignore
    load_from_disk = None  # type: ignore

from .prompt import build_user_prompt


@dataclass(frozen=True)
class PromptSample:
    """
    Container capturing a rendered prompt with associated metadata.

    :ivar issue: Issue label that the prompt pertains to.
    :vartype issue: str
    :ivar participant_study: Study identifier when available.
    :vartype participant_study: str | None
    :ivar participant_id: Participant identifier or ``None`` when omitted.
    :vartype participant_id: str | None
    :ivar split: Dataset split name (e.g. ``"train"``).
    :vartype split: str
    :ivar prompt: Rendered prompt text.
    :vartype prompt: str
    """

    issue: str
    participant_study: str | None
    participant_id: str | None
    split: str
    prompt: str


def _ensure_dataset() -> None:
    """
    Validate that the optional ``datasets`` dependency is available.

    :raises ImportError: If :mod:`datasets` is not installed.
    """
    if load_from_disk is None:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "datasets must be installed to generate prompt samples "
            "(pip install datasets)"
        )


def _iter_splits(dataset_dict: DatasetDict) -> Iterable[tuple[str, any]]:
    """Yield non-empty dataset splits and their names.

    :param dataset_dict: Dataset dictionary containing splits.
    :type dataset_dict: DatasetDict
    :returns: Iterator of ``(split_name, split_dataset)`` tuples.
    :rtype: Iterable[tuple[str, any]]
    """
    for split_name, split in dataset_dict.items():
        if split is None or len(split) == 0:
            continue
        yield split_name, split


def generate_prompt_samples(
    *,
    dataset_path: str,
    issues: Sequence[str],
    max_per_issue: int = 1,
    max_history: int = 8,
) -> List[PromptSample]:
    """Load prompts for the requested ``issues`` from ``dataset_path``.

    :param dataset_path: Location of the cleaned dataset (Hugging Face format).
    :type dataset_path: str
    :param issues: Iterable of issue labels to sample.
    :type issues: Sequence[str]
    :param max_per_issue: Number of prompts to collect per issue.
    :type max_per_issue: int
    :param max_history: History length passed to :func:`prompt_builder.prompt.build_user_prompt`.
    :type max_history: int
    :returns: List of :class:`PromptSample` instances.
    :rtype: List[PromptSample]
    """

    _ensure_dataset()
    dataset_dict = load_from_disk(dataset_path)
    if not isinstance(dataset_dict, DatasetDict):
        raise ValueError(f"Dataset at {dataset_path!r} is not a DatasetDict")

    samples: List[PromptSample] = []
    for issue in issues:
        selected: List[PromptSample] = []
        fallbacks: List[PromptSample] = []
        for split_name, split in _iter_splits(dataset_dict):
            for row in split:
                if str(row.get("issue") or "").strip() != issue:
                    continue
                prompt = build_user_prompt(row, max_hist=max_history).strip()
                sample = PromptSample(
                    issue=issue,
                    participant_study=row.get("participant_study"),
                    participant_id=row.get("participant_id"),
                    split=split_name,
                    prompt=prompt,
                )
                # Prefer samples where the prompt showcases a non-empty watch history.
                if "(no recently watched videos available)" in prompt:
                    if len(fallbacks) < max_per_issue:
                        fallbacks.append(sample)
                else:
                    selected.append(sample)
                if len(selected) >= max_per_issue:
                    break
            if len(selected) >= max_per_issue:
                break
        if len(selected) < max_per_issue:
            needed = max_per_issue - len(selected)
            selected.extend(fallbacks[:needed])
        samples.extend(selected)
    return samples


def _normalise_output_path(path: str) -> Path:
    """Return a validated Markdown output path, creating parent directories.

    :param path: Destination filepath provided via CLI.
    :type path: str
    :returns: Path object pointing to a Markdown file.
    :rtype: Path
    :raises ValueError: If the path is not Markdown.
    """
    output_path = Path(path)
    if output_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"Output file must be Markdown (.md); received {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _format_sample_heading(idx: int, sample: PromptSample) -> str:
    """
    Build a Markdown heading summarising prompt metadata.

    :param idx: One-based sample index.
    :type idx: int
    :param sample: Sample container with prompt details.
    :type sample: PromptSample
    :returns: Markdown heading string.
    :rtype: str
    """
    meta_parts = [
        f"Issue: {sample.issue.replace('_', ' ').title()}",
        f"Split: {sample.split}",
    ]
    if sample.participant_study:
        meta_parts.append(f"Study: {sample.participant_study}")
    return f"### Sample {idx} ({', '.join(meta_parts)})"


def write_samples_markdown(samples: Sequence[PromptSample], output_path: str) -> Path:
    """
    Write ``samples`` to ``output_path`` in a readable Markdown layout.

    :param samples: Iterable of prompt samples to render.
    :type samples: Sequence[PromptSample]
    :param output_path: Destination Markdown file.
    :type output_path: str
    :returns: Path to the written Markdown file.
    :rtype: Path
    """

    destination = _normalise_output_path(output_path)
    lines: List[str] = [
        "# Prompt Builder Sample Prompts",
        "",
        "The samples below are generated via `prompt_builder.build_user_prompt` "
        "using the cleaned GRAIL dataset.",
        "",
    ]
    for idx, sample in enumerate(samples, start=1):
        lines.append(_format_sample_heading(idx, sample))
        lines.append("")
        lines.append("```text")
        lines.append(sample.prompt)
        lines.append("```")
        lines.append("")
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


def _parse_issue_list(raw: str | None) -> List[str]:
    """
    Parse a comma-separated issue list into normalised tokens.

    :param raw: Raw issue string supplied via CLI or ``None``.
    :type raw: str | None
    :returns: List of individual issue identifiers.
    :rtype: List[str]
    """
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _build_cli() -> argparse.ArgumentParser:
    """Construct the CLI argument parser for the sample generator.

    :returns: Argument parser configured for the sample generator CLI.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Generate sample prompts for documentation."
    )
    parser.add_argument(
        "--dataset",
        default="data/cleaned_grail",
        help="Path to the cleaned dataset directory.",
    )
    parser.add_argument(
        "--issues",
        default="gun_control,minimum_wage",
        help="Comma-separated list of issues to sample.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1,
        help="Minimum number of prompts to include per issue.",
    )
    parser.add_argument(
        "--total",
        type=int,
        default=10,
        help="Total number of prompts to include across all issues.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=8,
        dest="max_history",
        help="History length passed to build_user_prompt.",
    )
    parser.add_argument(
        "--output",
        default="reports/prompt_builder/README.md",
        help="Destination Markdown file for the samples.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """
    Entry point for the prompt-sample CLI.

    :param argv: Optional list of CLI arguments; defaults to :data:`sys.argv`.
    :type argv: Sequence[str] | None
    :raises ValueError: If no issues are provided via ``--issues``.
    """
    parser = _build_cli()
    args = parser.parse_args(argv)
    issues = _parse_issue_list(args.issues)
    if not issues:
        raise ValueError("At least one issue must be specified via --issues.")
    total_requested = max(args.total, len(issues))
    per_issue = max(
        args.count,
        math.ceil(total_requested / len(issues)) if issues else total_requested,
    )
    samples = generate_prompt_samples(
        dataset_path=args.dataset,
        issues=issues,
        max_per_issue=max(per_issue, 1),
        max_history=max(args.max_history, 1),
    )
    if len(samples) > total_requested:
        samples = samples[:total_requested]
    destination = write_samples_markdown(samples, args.output)
    print(f"[prompt_builder] Wrote {len(samples)} samples to {destination}")


if __name__ == "__main__":  # pragma: no cover
    main()

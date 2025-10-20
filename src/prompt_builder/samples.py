"""Utilities for generating sample prompts for documentation and reports."""

from __future__ import annotations

import argparse
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
    """Container capturing a rendered prompt with minimal metadata."""

    issue: str
    participant_study: str | None
    participant_id: str | None
    split: str
    prompt: str


def _ensure_dataset() -> None:
    if load_from_disk is None:  # pragma: no cover - optional dependency guard
        raise ImportError(
            "datasets must be installed to generate prompt samples "
            "(pip install datasets)"
        )


def _iter_splits(ds: DatasetDict) -> Iterable[tuple[str, any]]:
    for split_name, split in ds.items():
        if split is None or not len(split):
            continue
        yield split_name, split


def _first_examples(
    ds: DatasetDict,
    *,
    issue: str,
    limit: int,
) -> List[dict]:
    """Return up to ``limit`` examples for ``issue`` across available splits."""

    collected: List[dict] = []
    for split_name, split in _iter_splits(ds):
        for row in split:
            if str(row.get("issue") or "").strip() != issue:
                continue
            collected.append((split_name, row))
            if len(collected) >= limit:
                return collected
    return collected


def generate_prompt_samples(
    *,
    dataset_path: str,
    issues: Sequence[str],
    max_per_issue: int = 1,
    max_history: int = 8,
) -> List[PromptSample]:
    """Load prompts for the requested ``issues`` from ``dataset_path``.

    :param dataset_path: Location of the cleaned dataset (Hugging Face format).
    :param issues: Iterable of issue labels to sample.
    :param max_per_issue: Number of prompts to collect per issue.
    :param max_history: History length passed to :func:`build_user_prompt`.
    :returns: List of :class:`PromptSample` instances.
    """

    _ensure_dataset()
    ds = load_from_disk(dataset_path)
    if not isinstance(ds, DatasetDict):
        raise ValueError(f"Dataset at {dataset_path!r} is not a DatasetDict")

    samples: List[PromptSample] = []
    for issue in issues:
        examples = _first_examples(ds, issue=issue, limit=max_per_issue)
        for split_name, row in examples:
            prompt = build_user_prompt(row, max_hist=max_history)
            samples.append(
                PromptSample(
                    issue=issue,
                    participant_study=row.get("participant_study"),
                    participant_id=row.get("participant_id"),
                    split=split_name,
                    prompt=prompt.strip(),
                )
            )
    return samples


def _normalise_output_path(path: str) -> Path:
    output_path = Path(path)
    if output_path.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError(f"Output file must be Markdown (.md); received {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def _format_sample_heading(idx: int, sample: PromptSample) -> str:
    meta_parts = [
        f"Issue: {sample.issue.replace('_', ' ').title()}",
        f"Split: {sample.split}",
    ]
    if sample.participant_study:
        meta_parts.append(f"Study: {sample.participant_study}")
    return f"### Sample {idx} ({', '.join(meta_parts)})"


def write_samples_markdown(samples: Sequence[PromptSample], output_path: str) -> Path:
    """Write ``samples`` to ``output_path`` in a readable Markdown layout."""

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
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _build_cli() -> argparse.ArgumentParser:
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
        help="Number of prompts to include per issue.",
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
        default="reports/prompt_builder/sample_prompts.md",
        help="Destination Markdown file for the samples.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_cli()
    args = parser.parse_args(argv)
    issues = _parse_issue_list(args.issues)
    if not issues:
        raise ValueError("At least one issue must be specified via --issues.")
    samples = generate_prompt_samples(
        dataset_path=args.dataset,
        issues=issues,
        max_per_issue=max(args.count, 1),
        max_history=max(args.max_history, 1),
    )
    destination = write_samples_markdown(samples, args.output)
    print(f"[prompt_builder] Wrote {len(samples)} samples to {destination}")


if __name__ == "__main__":  # pragma: no cover
    main()

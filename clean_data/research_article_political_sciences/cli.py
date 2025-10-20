"""Command-line entry point for the political sciences replication report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from clean_data.prompt.utils import load_dataset_any

from .report import generate_research_article_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate heatmaps and summaries replicating the PNAS filter-bubble "
            "study using the cleaned dataset."
        )
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Path to a dataset saved with Dataset.save_to_disk or a Hugging Face hub id.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where heatmaps and the Markdown report will be written.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=10,
        help="Number of equally spaced bins to use along each axis of the heatmaps.",
    )
    return parser.parse_args()


def main() -> Dict[str, Any]:
    """Entry point used by ``python -m`` to run the replication analysis."""

    args = _parse_args()
    dataset = load_dataset_any(args.dataset)
    return generate_research_article_report(
        dataset,
        output_dir=Path(args.output_dir),
        heatmap_bins=max(2, args.bins),
    )


__all__ = ["main"]


if __name__ == "__main__":
    main()

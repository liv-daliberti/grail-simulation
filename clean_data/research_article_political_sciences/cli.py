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

"""Command-line entry point for the political sciences replication report.

This CLI loads a cleaned dataset, orchestrates the opinion-shift analysis,
and writes the figures plus Markdown summary replicating the published
study. Usage of this entry point is covered by the repository's Apache 2.0
license; consult LICENSE for details.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

from clean_data.prompt.utils import load_dataset_any

from .report import generate_research_article_report


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the replication report entry point.

    :returns: Namespace containing dataset path, output directory, and bin count.
    """

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
    """Entry point used by ``python -m`` to run the replication analysis.

    :returns: Dictionary of report metadata returned by :func:`generate_research_article_report`.
    """

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

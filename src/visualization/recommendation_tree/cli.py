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

"""Command-line helpers for recommendation tree visualisations."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

from . import io, render
from .models import DEFAULT_LABEL_TEMPLATE, SESSION_DEFAULT_LABEL_TEMPLATE


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the recommendation visualiser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, help="Path to a tree CSV file.")
    parser.add_argument(
        "--cleaned-data",
        type=Path,
        help=(
            "Path to a cleaned dataset produced by clean_data/clean_data.py "
            "(datasets.save_to_disk)."
        ),
    )
    parser.add_argument(
        "--session-id",
        help=(
            "Session ID to visualize when using --cleaned-data. Defaults to the first "
            "session found."
        ),
    )
    parser.add_argument(
        "--split",
        help=(
            "Dataset split to read from when using --cleaned-data "
            "(defaults to every available split)."
        ),
    )
    parser.add_argument(
        "--issue",
        help="Optional issue filter when using --cleaned-data (e.g. gun_control or minimum_wage).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        help="Limit the number of recommendation steps rendered for the selected session.",
    )
    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        help="Directory to emit multiple session visualisations (requires --cleaned-data).",
    )
    parser.add_argument(
        "--batch-issues",
        default="minimum_wage=2,gun_control=2",
        help=(
            "Comma separated list of issue=count pairs when using --batch-output-dir. "
            "Defaults to minimum_wage=2,gun_control=2."
        ),
    )
    parser.add_argument(
        "--batch-prefix",
        default="session",
        help="Filename prefix when writing batch visualisations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output path (extension selects the format, e.g. .png or .svg).",
    )
    parser.add_argument(
        "--id-column",
        default="originId",
        help="Column that holds the video identifier inside the tree CSV.",
    )
    parser.add_argument(
        "--child-prefixes",
        default="rec",
        help="Comma separated prefixes used to detect recommendation columns.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional metadata CSV/JSON with extra information for labels.",
    )
    parser.add_argument(
        "--metadata-id-column",
        default="originId",
        help="Identifier column inside the metadata file.",
    )
    parser.add_argument(
        "--label-template",
        default=DEFAULT_LABEL_TEMPLATE,
        help=(
            "Python format string used to render node labels. "
            "Available fields include any column from the tree CSV or metadata."
        ),
    )
    parser.add_argument(
        "--wrap-width",
        type=int,
        default=30,
        help="Wrap labels to this many characters per line (set 0 to disable).",
    )
    parser.add_argument(
        "--highlight",
        help="Comma separated list of video IDs describing a viewer path to highlight.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        help=(
            "Optional path to trajectories (CSV, TXT, JSON, JSONL). "
            "Each row or entry should describe a sequence of video IDs."
        ),
    )
    parser.add_argument(
        "--trajectory-delimiter",
        default=",",
        help="Delimiter used when parsing plain-text trajectory files.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        help="Limit the depth of the rendered tree (root depth = 0).",
    )
    parser.add_argument(
        "--rankdir",
        default="LR",
        choices=["TB", "LR", "BT", "RL"],
        help="Graph orientation (Graphviz rankdir).",
    )
    parser.add_argument(
        "--engine",
        default="dot",
        help="Graphviz layout engine (dot, neato, fdp, etc.).",
    )
    parser.add_argument(
        "--hide-rank-labels",
        action="store_true",
        help="Do not annotate edges with their recommendation rank.",
    )
    parser.add_argument(
        "--format",
        help="Override the Graphviz output format (otherwise inferred from --output).",
    )
    args = parser.parse_args(argv)
    if not args.tree and not args.cleaned_data:
        parser.error("Please provide either --tree or --cleaned-data.")
    if not args.batch_output_dir and not args.output:
        parser.error("Please provide --output or --batch-output-dir.")
    return args


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point for the recommendation tree visualiser."""

    # pylint: disable=too-many-branches,too-many-locals,too-many-statements
    args = parse_args(argv)
    highlight_path: List[str] = []
    if args.highlight:
        highlight_path = [token.strip() for token in args.highlight.split(",") if token.strip()]

    dataset = io.load_cleaned_dataset(args.cleaned_data) if args.cleaned_data else None

    if args.batch_output_dir:
        if args.tree:
            raise SystemExit("--batch-output-dir currently supports --cleaned-data only.")
        if not dataset:
            raise SystemExit("--batch-output-dir requires --cleaned-data.")
        if args.issue:
            raise SystemExit(
                "--batch-output-dir is incompatible with --issue. Use --batch-issues instead."
            )
        try:
            issue_targets = io.parse_issue_counts(args.batch_issues)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not issue_targets:
            raise SystemExit("No issue counts provided for batch rendering.")
        output_dir = args.batch_output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_format = args.format or "svg"
        session_label_template = (
            SESSION_DEFAULT_LABEL_TEMPLATE
            if args.label_template == DEFAULT_LABEL_TEMPLATE
            else args.label_template
        )

        emitted = 0
        for issue, count in issue_targets.items():
            if count <= 0:
                continue
            rows = io.collect_rows(dataset, split=args.split, issue=issue)
            sessions = io.group_rows_by_session(rows)
            if len(sessions) < count:
                raise SystemExit(
                    f"Requested {count} session(s) for issue '{issue}', "
                    f"but only {len(sessions)} found."
                )
            for idx, session_id in enumerate(sorted(sessions)[:count], start=1):
                session_rows = sessions[session_id]
                if args.max_steps and args.max_steps > 0:
                    session_rows = session_rows[: args.max_steps]
                graph = render.build_session_graph(
                    session_rows,
                    label_template=session_label_template,
                    wrap_width=args.wrap_width,
                    rankdir=args.rankdir,
                    engine=args.engine,
                    highlight_path=highlight_path,
                )
                filename = f"{args.batch_prefix}_{issue}_{idx}.{output_format}"
                output_path = output_dir / filename
                render.render_graph(graph, output_path, output_format=output_format)
                print(f"Wrote {output_path}", file=sys.stderr)
                emitted += 1
        if emitted == 0:
            raise SystemExit(
                "No sessions rendered. Check --batch-issues counts and dataset filters."
            )
        return

    if args.cleaned_data:
        assert dataset is not None  # for type checkers
        session_id, session_rows = io.extract_session_rows(
            dataset,
            session_id=args.session_id,
            split=args.split,
            issue=args.issue,
            max_steps=args.max_steps,
        )
        session_label_template = (
            SESSION_DEFAULT_LABEL_TEMPLATE
            if args.label_template == DEFAULT_LABEL_TEMPLATE
            else args.label_template
        )
        graph = render.build_session_graph(
            session_rows,
            label_template=session_label_template,
            wrap_width=args.wrap_width,
            rankdir=args.rankdir,
            engine=args.engine,
            highlight_path=highlight_path,
        )
    else:
        child_prefixes = tuple(
            prefix.strip()
            for prefix in args.child_prefixes.split(",")
            if prefix.strip()
        )
        tree = io.load_tree_csv(args.tree, id_column=args.id_column, child_prefixes=child_prefixes)
        metadata = io.load_metadata(args.metadata, id_column=args.metadata_id_column)
        sequences = io.load_trajectories(args.trajectories, delimiter=args.trajectory_delimiter)
        if sequences:
            node_counts, edge_counts = render.aggregate_counts(sequences)
        else:
            node_counts, edge_counts = Counter(), Counter()
        graph = render.build_graph(
            tree,
            metadata=metadata,
            label_template=args.label_template,
            wrap_width=args.wrap_width,
            highlight_path=highlight_path,
            node_counts=node_counts,
            edge_counts=edge_counts,
            max_depth=args.max_depth,
            rankdir=args.rankdir,
            engine=args.engine,
            show_rank_labels=not args.hide_rank_labels,
        )
    output_format = args.format or (args.output.suffix.lstrip(".") if args.output else "")
    render.render_graph(graph, args.output, output_format=output_format)


__all__ = [
    "DEFAULT_LABEL_TEMPLATE",
    "SESSION_DEFAULT_LABEL_TEMPLATE",
    "main",
    "parse_args",
]

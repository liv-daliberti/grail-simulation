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
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple, List

from . import io, render
from .models import DEFAULT_LABEL_TEMPLATE, SESSION_DEFAULT_LABEL_TEMPLATE


@dataclass(frozen=True)
class BatchContext:
    """Configuration for batch session rendering."""

    dataset: object
    issue_targets: Mapping[str, int]
    output_dir: Path
    output_format: str
    label_template: str
    wrap_width: Optional[int]
    rankdir: str
    engine: str
    highlight_path: Sequence[str]
    split: Optional[str]
    max_steps: Optional[int]
    batch_prefix: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the recommendation visualiser.

    :param argv: Optional CLI argument list; defaults to ``sys.argv`` when ``None``.
    :returns: Parsed argument namespace ready for execution.
    """

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
    """Entry point for the recommendation tree visualiser.

    :param argv: Optional CLI argument list; defaults to ``sys.argv``.
    """

    args = parse_args(argv)
    highlight_path = tuple(_parse_highlight_argument(args.highlight))
    dataset = io.load_cleaned_dataset(args.cleaned_data) if args.cleaned_data else None

    if args.batch_output_dir:
        _render_batch_sessions(args, dataset, highlight_path)
        return

    graph = _build_single_graph(args, dataset, highlight_path)
    output_format = _resolve_output_format(args)
    render.render_graph(graph, args.output, output_format=output_format)


def _parse_highlight_argument(raw: Optional[str]) -> List[str]:
    """Normalise the comma-separated highlight argument into identifiers.

    :param raw: Raw comma-separated string supplied via ``--highlight``.
    :returns: List of cleaned identifiers.
    """

    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]


def _build_single_graph(
    args: argparse.Namespace,
    dataset,
    highlight_path: Sequence[str],
):
    """Construct the graph for either cleaned datasets or CSV inputs.

    :param args: Parsed CLI arguments.
    :param dataset: Loaded cleaned dataset when ``--cleaned-data`` is used.
    :param highlight_path: Sequence of identifiers to highlight.
    :returns: Graphviz graph ready to render.
    """

    if args.cleaned_data:
        if dataset is None:
            raise SystemExit("Unable to load the cleaned dataset from the provided path.")
        return _session_graph_from_dataset(args, dataset, highlight_path)
    return _tree_graph_from_files(args, highlight_path)


def _session_graph_from_dataset(
    args: argparse.Namespace,
    dataset,
    highlight_path: Sequence[str],
):
    """Build a session graph for a single viewer trajectory.

    :param args: Parsed CLI arguments.
    :param dataset: Loaded cleaned dataset object.
    :param highlight_path: Sequence of identifiers to highlight.
    :returns: Graphviz graph representing a session.
    """

    _, session_rows = io.extract_session_rows(
        dataset,
        session_id=args.session_id,
        split=args.split,
        issue=args.issue,
        max_steps=args.max_steps,
    )
    label_template = _session_label_template(args)
    return _build_session_graph_for_rows(
        session_rows,
        label_template,
        args,
        highlight_path,
    )


def _tree_graph_from_files(
    args: argparse.Namespace,
    highlight_path: Sequence[str],
):
    """Build a tree graph directly from CSV exports.

    :param args: Parsed CLI arguments.
    :param highlight_path: Sequence of identifiers to highlight.
    :returns: Graphviz graph representing the tree export.
    """

    child_prefixes = _parse_child_prefixes(args.child_prefixes)
    tree = io.load_tree_csv(args.tree, id_column=args.id_column, child_prefixes=child_prefixes)
    metadata = io.load_metadata(args.metadata, id_column=args.metadata_id_column)
    sequences = io.load_trajectories(args.trajectories, delimiter=args.trajectory_delimiter)
    if sequences:
        node_counts, edge_counts = render.aggregate_counts(sequences)
    else:
        node_counts, edge_counts = Counter(), Counter()
    options = render.GraphRenderOptions(
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
    return render.build_graph(tree, options)


def _build_session_graph_for_rows(
    rows,
    label_template: str,
    args: argparse.Namespace,
    highlight_path: Sequence[str],
):
    """Render a session graph from pre-selected rows.

    :param rows: Dataset rows for the target session.
    :param label_template: Template string used for node labels.
    :param args: Parsed CLI arguments to reuse other options.
    :param highlight_path: Sequence of identifiers to highlight.
    :returns: Graphviz graph representing the session.
    """

    options = render.SessionGraphOptions(
        label_template=label_template,
        wrap_width=args.wrap_width,
        rankdir=args.rankdir,
        engine=args.engine,
        highlight_path=highlight_path,
    )
    return render.build_session_graph(rows, options)


def _session_label_template(args: argparse.Namespace) -> str:
    """Return the appropriate label template for session renders.

    :param args: Parsed CLI arguments.
    :returns: Template string to use when rendering session nodes.
    """

    if args.label_template == DEFAULT_LABEL_TEMPLATE:
        return SESSION_DEFAULT_LABEL_TEMPLATE
    return args.label_template


def _parse_child_prefixes(raw: str) -> Tuple[str, ...]:
    """Split and normalise child prefix specifications.

    :param raw: Comma-separated list provided via ``--child-prefixes``.
    :returns: Tuple of cleaned prefixes, defaulting to ``(\"rec\",)`` if empty.
    """

    prefixes = tuple(prefix.strip() for prefix in raw.split(",") if prefix.strip())
    return prefixes or ("rec",)


def _resolve_output_format(args: argparse.Namespace) -> str:
    """Determine the user-requested output format.

    :param args: Parsed CLI arguments.
    :returns: Normalised Graphviz format name or an empty string.
    """

    if args.format:
        return args.format
    if args.output:
        return args.output.suffix.lstrip(".")
    return ""


def _render_batch_sessions(
    args: argparse.Namespace,
    dataset,
    highlight_path: Sequence[str],
) -> None:
    """Render multiple sessions organised by issue into an output directory.

    :param args: Parsed CLI arguments.
    :param dataset: Loaded cleaned dataset object.
    :param highlight_path: Sequence of identifiers to highlight.
    """

    context = _prepare_batch_context(args, dataset, highlight_path)
    emitted = 0
    for issue, count in context.issue_targets.items():
        emitted += _render_issue_sessions(issue, count, context)
    if emitted == 0:
        raise SystemExit(
            "No sessions rendered. Check --batch-issues counts and dataset filters."
        )


def _prepare_batch_context(
    args: argparse.Namespace,
    dataset,
    highlight_path: Sequence[str],
) -> BatchContext:
    """Validate batch arguments and assemble a rendering context.

    :param args: Parsed CLI arguments.
    :param dataset: Loaded cleaned dataset object.
    :param highlight_path: Sequence of identifiers to highlight.
    :returns: Batch rendering configuration.
    """

    if args.tree:
        raise SystemExit("--batch-output-dir currently supports --cleaned-data only.")
    if dataset is None:
        raise SystemExit("--batch-output-dir requires --cleaned-data.")
    if args.issue:
        raise SystemExit("--batch-output-dir is incompatible with --issue. Use --batch-issues instead.")
    try:
        issue_targets = io.parse_issue_counts(args.batch_issues)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not issue_targets:
        raise SystemExit("No issue counts provided for batch rendering.")
    output_dir = args.batch_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return BatchContext(
        dataset=dataset,
        issue_targets=dict(issue_targets),
        output_dir=output_dir,
        output_format=args.format or "svg",
        label_template=_session_label_template(args),
        wrap_width=args.wrap_width,
        rankdir=args.rankdir,
        engine=args.engine,
        highlight_path=tuple(highlight_path),
        split=args.split,
        max_steps=args.max_steps,
        batch_prefix=args.batch_prefix,
    )


def _render_issue_sessions(issue: str, count: int, context: BatchContext) -> int:
    """Render one issue's worth of sessions, returning the emitted graph count.

    :param issue: Dataset issue identifier (e.g. ``minimum_wage``).
    :param count: Number of sessions to render for the issue.
    :param context: Batch rendering configuration.
    :returns: Number of emitted graphs for the issue.
    """

    if count <= 0:
        return 0
    rows = io.collect_rows(context.dataset, split=context.split, issue=issue)
    sessions = io.group_rows_by_session(rows)
    if len(sessions) < count:
        raise SystemExit(
            f"Requested {count} session(s) for issue '{issue}', but only {len(sessions)} found."
        )
    emitted = 0
    for idx, session_id in enumerate(sorted(sessions)[:count], start=1):
        session_rows = list(sessions[session_id])
        if context.max_steps and context.max_steps > 0:
            session_rows = session_rows[: context.max_steps]
        graph = render.build_session_graph(
            session_rows,
            render.SessionGraphOptions(
                label_template=context.label_template,
                wrap_width=context.wrap_width,
                rankdir=context.rankdir,
                engine=context.engine,
                highlight_path=context.highlight_path,
            ),
        )
        filename = f"{context.batch_prefix}_{issue}_{idx}.{context.output_format}"
        output_path = context.output_dir / filename
        render.render_graph(graph, output_path, output_format=context.output_format)
        print(f"Wrote {output_path}", file=sys.stderr)
        emitted += 1
    return emitted


__all__ = [
    "DEFAULT_LABEL_TEMPLATE",
    "SESSION_DEFAULT_LABEL_TEMPLATE",
    "main",
    "parse_args",
]

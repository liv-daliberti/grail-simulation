"""Tests for the recommendation tree visualisation helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping

import pytest

from src.visualization.recommendation_tree_viz import (
    TreeData,
    TreeEdge,
    aggregate_counts,
    build_graph,
    build_session_graph,
    compute_depths,
    format_node_label,
    load_trajectories,
    parse_issue_counts,
    render_graph,
)

pytestmark = pytest.mark.visualization


def test_parse_issue_counts_parses_input() -> None:
    """Issue specifications should produce integer counts."""
    result = parse_issue_counts("gun_control=2, minimum_wage=1")
    assert result == {"gun_control": 2, "minimum_wage": 1}


def test_parse_issue_counts_rejects_invalid_input() -> None:
    """Invalid count specifications should raise an error."""
    with pytest.raises(ValueError):
        parse_issue_counts("no-count")


def test_compute_depths_and_aggregate_counts() -> None:
    """Depth and count helpers should reflect simple tree structures."""
    edges = [
        TreeEdge("root", "child_a"),
        TreeEdge("root", "child_b"),
        TreeEdge("child_a", "leaf"),
    ]
    tree = TreeData(
        root="root",
        nodes={"root": {}, "child_a": {}, "child_b": {}, "leaf": {}},
        edges=edges,
    )
    depths = compute_depths(tree)
    assert depths == {"root": 0, "child_a": 1, "child_b": 1, "leaf": 2}

    sequences = [["root", "child_a", "leaf"], ["root", "child_b"]]
    node_counts, edge_counts = aggregate_counts(sequences)
    assert node_counts == Counter({"root": 2, "child_a": 1, "leaf": 1, "child_b": 1})
    assert edge_counts == Counter(
        {("root", "child_a"): 1, ("child_a", "leaf"): 1, ("root", "child_b"): 1}
    )


def test_format_node_label_and_build_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tree graphs should honour metadata, counts, and highlight settings."""
    nodes: Dict[str, Mapping[str, object]] = {
        "root": {"originTitle": "Root Node"},
        "child": {"originTitle": "Child Node"},
    }
    edges = [TreeEdge("root", "child", rank=1)]
    tree = TreeData(root="root", nodes=nodes, edges=edges)
    metadata = {"root": {"title": "Root Node"}, "child": {"title": "Child Node"}}
    node_counts = Counter({"root": 3, "child": 1})
    edge_counts = Counter({("root", "child"): 1})

    graph = build_graph(
        tree,
        metadata=metadata,
        label_template="{title}",
        wrap_width=None,
        highlight_path=("root", "child"),
        node_counts=node_counts,
        edge_counts=edge_counts,
        max_depth=None,
        rankdir="LR",
        engine="dot",
        show_rank_labels=True,
    )

    source = graph.source
    assert 'xlabel=3' in source
    assert 'fillcolor="#81a1c1"' in source  # highlight applies
    assert "Rec 1" in source  # rank labels included

    output_path = tmp_path / "graph.svg"
    label = format_node_label(
        "child",
        node_data=nodes["child"],
        metadata=metadata,
        template="{title}",
        wrap_width=10,
    )
    assert label.startswith("Child")
    # Rendering should succeed and place a file at the expected location.

    def _fake_render(*, filename: str, cleanup: bool) -> str:
        del cleanup  # parameter unused in fake
        target = tmp_path / f"{Path(filename).name}.{graph.format}"
        target.write_text("graph", encoding="utf-8")
        return str(target)

    monkeypatch.setattr(graph, "render", _fake_render)
    rendered_path = render_graph(graph, output_path, output_format="svg")
    assert rendered_path == output_path
    assert output_path.exists()


def test_build_session_graph_handles_highlights() -> None:
    """Session graphs should mark chosen recommendations and highlighted nodes."""
    rows = [
        {
            "display_step": 1,
            "current_video_id": "v1",
            "current_video_title": "Video 1",
            "slate_items_json": [
                {"id": "v2", "title": "Video 2"},
                {"id": "v3", "title": "Video 3"},
            ],
            "next_video_id": "v2",
        },
        {
            "display_step": 2,
            "current_video_id": "v2",
            "current_video_title": "Video 2",
            "slate_items_json": [],
        },
    ]

    graph = build_session_graph(
        rows,
        label_template="{title}",
        wrap_width=None,
        rankdir="LR",
        engine="dot",
        highlight_path=("v2",),
    )

    source = graph.source
    assert "step1_opt1" in source  # chosen option node exists
    assert "label=selected" in source  # selected edge annotated
    assert '#bf616a' in source  # highlight colour applied


def test_load_trajectories_jsonl(tmp_path: Path) -> None:
    """Trajectory loader should normalise repeated identifiers."""
    path = tmp_path / "trajectories.jsonl"
    entries = [
        {"sequences": [["vid1", "vid1", "vid2"]]},
        {"sequences": [["vid3", "vid3", "vid4"]]},
    ]
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry) + "\n")

    trajectories = load_trajectories(path)
    assert trajectories == [["vid1", "vid2"], ["vid3", "vid4"]]

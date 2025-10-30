"""Tests for the recommendation tree visualisation helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Mapping

import pytest

from src.visualization.recommendation_tree_viz import (
    LabelRenderOptions,
    TreeData,
    TreeEdge,
    _extract_sequences_from_object,
    _extract_opinion_annotation,
    aggregate_counts,
    build_graph,
    build_session_graph,
    format_node_label,
    _opinion_label,
    compute_depths,
    load_metadata,
    load_trajectories,
    load_tree_csv,
    main,
    parse_args,
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


def test_opinion_label_negative_delta() -> None:
    """Opinion labels should preserve negative changes."""
    label = _opinion_label("Minimum wage support", "Final", 3.0, delta=-1.25)
    assert label == "Final Minimum wage support: 3 (-1.25)"


def test_extract_opinion_annotation_handles_selected_rows() -> None:
    """Opinion annotations should combine main rows with selected survey rows."""
    rows = [
        {
            "issue": "minimum_wage",
            "selected_survey_row": {
                "mw_index_w1": 4,
            },
        },
        {
            "issue": "minimum_wage",
            "selected_survey_row": {
                "mw_index_w2": 5,
            },
        },
    ]
    annotation = _extract_opinion_annotation(rows)
    assert annotation is not None
    assert annotation.before_value == 4
    assert annotation.after_value == 5


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
        options=LabelRenderOptions(metadata=metadata, template="{title}", wrap_width=10),
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


def test_build_session_graph_shows_negative_support_change() -> None:
    """Session graphs should surface decreases in issue support."""
    rows = [
        {
            "display_step": 1,
            "current_video_id": "v1",
            "current_video_title": "Video 1",
            "slate_items_json": [],
            "issue": "minimum_wage",
            "selected_survey_row": {
                "mw_index_w1": 5,
                "mw_index_w2": 3,
            },
        },
    ]
    graph = build_session_graph(
        rows,
        label_template="{title}",
        wrap_width=None,
        rankdir="LR",
        engine="dot",
        highlight_path=(),
    )
    source = graph.source
    assert "Initial Minimum wage support: 5" in source
    assert "Final Minimum wage support: 3 (-2)" in source


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


def test_load_tree_csv_detects_columns_and_edges(tmp_path: Path) -> None:
    """Tree CSV ingestion should detect identifier column and child prefixes."""
    csv_path = tmp_path / "tree.csv"
    csv_path.write_text(
        "\n".join(
            [
                "originid,rec1,rec2",
                "root,child,",
                "child,,",
            ]
        ),
        encoding="utf-8",
    )

    tree = load_tree_csv(csv_path)

    assert tree.root == "root"
    assert any(edge == TreeEdge(parent="root", child="child", rank=1) for edge in tree.edges)
    assert "root" in tree.nodes and tree.nodes["root"]["rec1"] == "child"
    assert "child" in tree.nodes


def test_load_tree_csv_requires_recommendation_columns(tmp_path: Path) -> None:
    """Missing recommendation columns should raise an informative error."""
    csv_path = tmp_path / "tree.csv"
    csv_path.write_text("originId\nroot\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_tree_csv(csv_path)


def test_load_metadata_supports_json_structures(tmp_path: Path) -> None:
    """Metadata loader should accept JSON payloads and normalise identifiers."""
    json_path = tmp_path / "meta.json"
    json_path.write_text(
        json.dumps(
            [
                {"originId": "root", "title": "Root Title"},
                {"originId": "child", "title": "Child Title"},
            ]
        ),
        encoding="utf-8",
    )

    metadata = load_metadata(json_path)
    assert metadata["root"]["title"] == "Root Title"
    assert metadata["child"]["title"] == "Child Title"


def test_load_metadata_supports_csv(tmp_path: Path) -> None:
    """Metadata loader should read CSV files with the configured identifier column."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "\n".join(
            [
                "originId,title",
                "root,Root Title",
                "child,Child Title",
            ]
        ),
        encoding="utf-8",
    )

    metadata = load_metadata(csv_path)
    assert metadata == {
        "root": {"originId": "root", "title": "Root Title"},
        "child": {"originId": "child", "title": "Child Title"},
    }


def test_load_metadata_missing_identifier_column(tmp_path: Path) -> None:
    """CSV metadata without the identifier column should raise a ValueError."""
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("title\nRoot Title\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_metadata(csv_path)


def test_extract_sequences_from_object_handles_variants() -> None:
    """Sequence extractor should cover list, mapping, and string payloads."""
    list_payload = [["vid1", "", "vid2"], ("vid3", "vid4")]
    mapping_payload = {"sequence": [["vid5", "vid6"], ["vid7"]]}
    string_payload = "vid8, vid9 , vid10"

    sequences_from_list = _extract_sequences_from_object(list_payload)
    sequences_from_mapping = _extract_sequences_from_object(mapping_payload)
    sequences_from_string = _extract_sequences_from_object(string_payload)

    assert sequences_from_list == [["vid1", "vid2"], ["vid3", "vid4"]]
    assert sequences_from_mapping == [["vid5", "vid6"], ["vid7"]]
    assert sequences_from_string == [["vid8", "vid9", "vid10"]]


def test_parse_args_converts_types(tmp_path: Path) -> None:
    """CLI parser should coerce path-like and numeric arguments correctly."""
    tree_csv = tmp_path / "tree.csv"
    output_svg = tmp_path / "out.svg"

    args = parse_args([
        "--tree",
        str(tree_csv),
        "--output",
        str(output_svg),
        "--wrap-width",
        "42",
    ])

    assert args.tree == tree_csv
    assert args.output == output_svg
    assert args.wrap_width == 42


def test_main_invokes_build_and_render(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Main entry point should load CSV/metadata and emit a rendered graph."""
    csv_path = tmp_path / "tree.csv"
    csv_path.write_text(
        "\n".join(
            [
                "originId,rec1",
                "root,child",
                "child,",
            ]
        ),
        encoding="utf-8",
    )

    metadata_path = tmp_path / "meta.json"
    metadata_path.write_text(
        json.dumps(
            [
                {"originId": "root", "title": "Root Title"},
                {"originId": "child", "title": "Child Title"},
            ]
        ),
        encoding="utf-8",
    )

    captured = {}

    def _fake_build_graph(
        tree,
        *,
        metadata,
        label_template,
        wrap_width,
        highlight_path,
        node_counts,
        edge_counts,
        max_depth,
        rankdir,
        engine,
        show_rank_labels,
    ):
        del (
            label_template,
            wrap_width,
            highlight_path,
            node_counts,
            edge_counts,
            max_depth,
            rankdir,
            engine,
            show_rank_labels,
        )

        captured["tree"] = tree
        captured["metadata"] = metadata

        class _FakeGraph:
            format = "svg"

            def render(self, filename: str, cleanup: bool) -> str:
                del cleanup
                target = tmp_path / f"{Path(filename).name}.svg"
                target.write_text("graph", encoding="utf-8")
                return str(target)

        return _FakeGraph()

    monkeypatch.setattr("src.visualization.recommendation_tree.cli.render.build_graph", _fake_build_graph)

    call_log = {}

    def _fake_render_graph(graph, output_path: Path, *, output_format: str) -> Path:
        call_log["graph"] = graph
        call_log["output_path"] = output_path
        call_log["output_format"] = output_format
        return output_path

    monkeypatch.setattr("src.visualization.recommendation_tree.cli.render.render_graph", _fake_render_graph)

    output_path = tmp_path / "rendered.svg"

    main(
        [
            "--tree",
            str(csv_path),
            "--metadata",
            str(metadata_path),
            "--output",
            str(output_path),
        ]
    )

    assert captured["tree"].root == "root"
    assert set(captured["metadata"]) == {"root", "child"}
    assert call_log["output_path"] == output_path
    assert call_log["output_format"] == "svg"

"""Tests for slate construction helpers."""

import pytest

from clean_data.sessions import build_slate_items, normalize_display_orders


pytestmark = pytest.mark.sessions

def test_build_slate_items_prefers_display_orders():
    display_orders = normalize_display_orders({"0": ["vidA", "vidB"]})
    tree_meta = {
        "vidA": {"title": "Video A", "channel_title": "Channel A", "channel_id": "chanA"},
        "vidB": {"title": "Video B"},
    }
    fallback_titles = {"vidB": "Video B Fallback"}
    recommendations = [{"id": "vidA"}, {"id": "vidB"}]

    items, source = build_slate_items(0, display_orders, recommendations, tree_meta, fallback_titles)

    assert source == "display_orders"
    assert [item["id"] for item in items] == ["vidA", "vidB"]
    assert items[0]["title"] == "Video A"
    assert items[1]["title"] == "Video B"


def test_build_slate_items_falls_back_to_recommendations_when_display_orders_missing():
    display_orders = normalize_display_orders({})
    tree_meta = {}
    fallback_titles = {"vidC": "Video C"}
    recommendations = [{"id": "vidC", "raw_id": "rawC"}]

    items, source = build_slate_items(1, display_orders, recommendations, tree_meta, fallback_titles)

    assert source == "tree_metadata"
    assert len(items) == 1
    assert items[0]["id"] == "vidC"
    assert items[0]["title"] == "Video C"


def test_build_slate_items_copies_engagement_stats_from_metadata():
    display_orders = normalize_display_orders({"0": ["vidD"]})
    tree_meta = {
        "vidD": {
            "title": "Video D",
            "view_count": 1234,
            "like_count": 567,
            "dislike_count": 8,
            "favorite_count": 9,
            "comment_count": 10,
            "duration": 321,
        }
    }
    fallback_titles = {}
    recommendations = [{"id": "vidD"}]

    items, source = build_slate_items(0, display_orders, recommendations, tree_meta, fallback_titles)

    assert source == "display_orders"
    assert items[0]["view_count"] == 1234
    assert items[0]["like_count"] == 567
    assert items[0]["dislike_count"] == 8
    assert items[0]["favorite_count"] == 9
    assert items[0]["comment_count"] == 10
    assert items[0]["duration"] == 321

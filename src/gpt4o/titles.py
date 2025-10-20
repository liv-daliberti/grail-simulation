"""Utilities for resolving slate video ids to human-readable titles."""

from __future__ import annotations

import csv
import os
from glob import glob
from typing import Dict, Iterable, Optional, Tuple

from .config import DEFAULT_TITLE_DIRS
from .utils import canon_video_id, split_env_list


def _guess_cols(header: Iterable[str]) -> Tuple[str | None, str | None]:
    """Guess the id/title column names within a CSV file."""

    candidates_id = [
        "originId",
        "ytid",
        "video_id",
        "youtube_id",
        "videoId",
        "origin_id",
        "id",
    ]
    candidates_title = ["originTitle", "title", "video_title", "name"]
    lower = {col.lower(): col for col in header}
    id_col = next((lower[item.lower()] for item in candidates_id if item.lower() in lower), None)
    title_col = next(
        (lower[item.lower()] for item in candidates_title if item.lower() in lower),
        None,
    )
    return id_col, title_col


class TitleResolver:
    """Resolve YouTube ids to titles by scanning configured CSV sources."""

    def __init__(self) -> None:
        self._index: Dict[str, str] | None = None

    def _iter_candidate_paths(self) -> list[str]:
        files: list[str] = []

        # Directories to crawl for CSV files
        for directory in split_env_list(os.environ.get("GRAIL_TITLE_DIRS")):
            if os.path.isdir(directory):
                for root, _, filenames in os.walk(directory):
                    for name in filenames:
                        if name.lower().endswith(".csv"):
                            files.append(os.path.join(root, name))

        # Glob expressions
        for pattern in split_env_list(os.environ.get("GRAIL_TITLE_GLOB")):
            files.extend([path for path in glob(pattern) if path.lower().endswith(".csv")])

        # Explicit CSV list
        for path in split_env_list(os.environ.get("GRAIL_TITLE_CSVS")):
            if os.path.isfile(path) and path.lower().endswith(".csv"):
                files.append(path)

        # Default directories (always scanned)
        for directory in DEFAULT_TITLE_DIRS:
            if os.path.isdir(directory):
                for root, _, filenames in os.walk(directory):
                    for name in filenames:
                        if name.lower().endswith(".csv"):
                            files.append(os.path.join(root, name))

        seen: set[str] = set()
        ordered: list[str] = []
        for path in files:
            if path not in seen:
                seen.add(path)
                ordered.append(path)
        return ordered

    def _build_index(self) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for path in self._iter_candidate_paths():
            try:
                with open(path, "r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    if not reader.fieldnames:
                        continue
                    id_col, title_col = _guess_cols(reader.fieldnames)
                    if not id_col or not title_col:
                        continue
                    for row in reader:
                        video_id = canon_video_id(row.get(id_col, "") or "")
                        title = (row.get(title_col, "") or "").strip()
                        if video_id and title and video_id not in index:
                            index[video_id] = title
            except Exception:
                continue
        return index

    def resolve(self, video_id: str | None) -> Optional[str]:
        """Return a title for the provided video id when available."""

        if not video_id:
            return None
        if self._index is None:
            self._index = self._build_index()
            print(f"[title-index] loaded {len(self._index)} titles from CSV")
        return self._index.get(canon_video_id(video_id))

    def __call__(self, video_id: str | None) -> Optional[str]:
        return self.resolve(video_id)


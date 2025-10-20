"""Shared helpers for resolving slate video ids to human-readable titles."""

from __future__ import annotations

import csv
import os
from glob import glob
from typing import Dict, Iterable, Optional, Sequence, Tuple

from .logging_utils import get_logger
from .text import canon_video_id, split_env_list

_DEFAULT_ENV_CSVS = "GRAIL_TITLE_CSVS"
_DEFAULT_ENV_DIRS = "GRAIL_TITLE_DIRS"
_DEFAULT_ENV_GLOB = "GRAIL_TITLE_GLOB"

_CANDIDATE_IDS = [
    "originId",
    "ytid",
    "video_id",
    "youtube_id",
    "videoId",
    "origin_id",
    "id",
]
_CANDIDATE_TITLES = ["originTitle", "title", "video_title", "name"]


def _guess_cols(header: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """Guess the id/title column names within a CSV file."""

    lower = {column.lower(): column for column in header}
    id_col = next((lower[item.lower()] for item in _CANDIDATE_IDS if item.lower() in lower), None)
    title_col = next(
        (lower[item.lower()] for item in _CANDIDATE_TITLES if item.lower() in lower),
        None,
    )
    return id_col, title_col


class TitleResolver:
    """Resolve YouTube ids to titles by scanning configured CSV sources."""

    def __init__(
        self,
        *,
        default_dirs: Sequence[str] | None = None,
        env_csv_var: str = _DEFAULT_ENV_CSVS,
        env_dirs_var: str = _DEFAULT_ENV_DIRS,
        env_glob_var: str = _DEFAULT_ENV_GLOB,
        logger_name: str = "title-index",
    ) -> None:
        """Create a new :class:`TitleResolver` instance.

        :param default_dirs: Optional list of fallback directories searched for CSVs.
        :param env_csv_var: Environment variable listing explicit CSV files.
        :param env_dirs_var: Environment variable listing directories to traverse.
        :param env_glob_var: Environment variable listing glob patterns for CSVs.
        :param logger_name: Logger name used for diagnostic output.
        """

        self._default_dirs = list(default_dirs or [])
        self._env_csv_var = env_csv_var
        self._env_dirs_var = env_dirs_var
        self._env_glob_var = env_glob_var
        self._index: Dict[str, str] | None = None
        self._logger = get_logger(logger_name)

    def _iter_candidate_paths(self) -> list[str]:
        """Enumerate all CSV files that might contain title metadata."""

        # pylint: disable=too-many-branches
        files: list[str] = []

        for directory in split_env_list(os.environ.get(self._env_dirs_var)):
            if os.path.isdir(directory):
                for root, _, filenames in os.walk(directory):
                    for name in filenames:
                        if name.lower().endswith(".csv"):
                            files.append(os.path.join(root, name))

        for pattern in split_env_list(os.environ.get(self._env_glob_var)):
            try:
                files.extend([path for path in glob(pattern) if path.lower().endswith(".csv")])
            except OSError:  # pragma: no cover - defensive
                continue

        for path in split_env_list(os.environ.get(self._env_csv_var)):
            if os.path.isfile(path) and path.lower().endswith(".csv"):
                files.append(path)

        for directory in self._default_dirs:
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
        """Construct the in-memory mapping from video id to title.

        :returns: Dictionary keyed by canonical video id.
        """

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
            except (OSError, UnicodeDecodeError, csv.Error):  # pragma: no cover - defensive
                continue
        self._logger.info("[title-index] loaded %d titles from CSV", len(index))
        return index

    def resolve(self, video_id: str | None) -> Optional[str]:
        """Return a title for the provided video id when available."""

        if not video_id:
            return None
        if self._index is None:
            self._index = self._build_index()
        return self._index.get(canon_video_id(video_id))

    def __call__(self, video_id: str | None) -> Optional[str]:
        """Proxy to :meth:`resolve` allowing instances to be callable."""

        return self.resolve(video_id)


__all__ = ["TitleResolver"]

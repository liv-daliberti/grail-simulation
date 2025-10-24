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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
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


@dataclass(frozen=True)
class TitleResolverEnvConfig:
    """

    Environment variable names used when sourcing title CSVs.



    :ivar csv_var: Attribute ``csv_var``.

    :vartype csv_var: str

    :ivar dirs_var: Attribute ``dirs_var``.

    :vartype dirs_var: str

    :ivar glob_var: Attribute ``glob_var``.

    :vartype glob_var: str

    """


    csv_var: str = _DEFAULT_ENV_CSVS
    dirs_var: str = _DEFAULT_ENV_DIRS
    glob_var: str = _DEFAULT_ENV_GLOB


def _guess_cols(header: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """

    Guess the id/title column names within a CSV file.



    :param header: Value provided for ``header``.

    :type header: Iterable[str]

    :returns: Result produced by ``_guess_cols``.

    :rtype: Tuple[Optional[str], Optional[str]]

    """


    lower = {column.lower(): column for column in header}
    id_col = next((lower[item.lower()] for item in _CANDIDATE_IDS if item.lower() in lower), None)
    title_col = next(
        (lower[item.lower()] for item in _CANDIDATE_TITLES if item.lower() in lower),
        None,
    )
    return id_col, title_col


class TitleResolver:
    """
    Resolve YouTube ids to titles by scanning configured CSV sources.

    :ivar _default_dirs: Fallback directories searched for metadata CSV files.
    :vartype _default_dirs: List[str]
    :ivar _env_csv_var: Environment variable that lists explicit CSV paths.
    :vartype _env_csv_var: str
    :ivar _env_dirs_var: Environment variable enumerating directories to search.
    :vartype _env_dirs_var: str
    :ivar _env_glob_var: Environment variable providing glob patterns.
    :vartype _env_glob_var: str
    :ivar _index: Cached mapping of video ids to titles once loaded.
    :vartype _index: Dict[str, str] | None
    :ivar _logger: Logger instance used for diagnostic output.
    :vartype _logger: logging.Logger
    """


    def __init__(
        self,
        *,
        default_dirs: Sequence[str] | None = None,
        env: TitleResolverEnvConfig | None = None,
        logger_name: str = "title-index",
    ) -> None:
        """Create a new :class:`TitleResolver` instance.

        :param default_dirs: Optional list of fallback directories searched for CSVs.
        :param env: Environment-variable configuration used while discovering CSVs.
        :param logger_name: Logger name used for diagnostic output.
        """

        config = env or TitleResolverEnvConfig()
        self._default_dirs = list(default_dirs or [])
        self._env_csv_var = config.csv_var
        self._env_dirs_var = config.dirs_var
        self._env_glob_var = config.glob_var
        self._index: Dict[str, str] | None = None
        self._logger = get_logger(logger_name)

    def _iter_candidate_paths(self) -> list[str]:
        """

        Enumerate all CSV files that might contain title metadata.



        :returns: Result produced by ``_iter_candidate_paths``.

        :rtype: list[str]

        """


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
        """

        Return a title for the provided video id when available.



        :param video_id: Value provided for ``video_id``.

        :type video_id: str | None

        :returns: Result produced by ``resolve``.

        :rtype: Optional[str]

        """


        if not video_id:
            return None
        if self._index is None:
            self._index = self._build_index()
        return self._index.get(canon_video_id(video_id))

    def __call__(self, video_id: str | None) -> Optional[str]:
        """

        Proxy to :meth:`resolve` allowing instances to be callable.



        :param video_id: Value provided for ``video_id``.

        :type video_id: str | None

        :returns: Result produced by ``__call__``.

        :rtype: Optional[str]

        """


        return self.resolve(video_id)


__all__ = ["TitleResolver", "TitleResolverEnvConfig"]

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

"""Discovery helpers for cached Word2Vec artefacts used across sweeps."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Callable, List, Sequence

from .settings import PipelinePaths

LOGGER = logging.getLogger("xgb.pipeline")

_WORD2VEC_WORKER_PATTERN = re.compile(r"training model with (\d+)\s+workers")
_VECTORISER_META = "vectorizer.json"


def _record_workers_from_dirs(
    directories: Sequence[Path], record: Callable[[int], None]
) -> None:
    """Scan cached vectoriser metadata files and record worker counts."""

    for base in directories:
        if base is None or not base.exists():
            continue
        try:
            iterator = base.rglob(_VECTORISER_META)
        except (OSError, RuntimeError):
            LOGGER.debug("Skipping Word2Vec metadata scan under %s.", base)
            continue
        for meta_path in iterator:
            try:
                with meta_path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError, UnicodeDecodeError):
                LOGGER.debug("Unable to parse vectoriser metadata at %s.", meta_path)
                continue
            config = payload.get("config") if isinstance(payload, dict) else None
            workers = config.get("workers") if isinstance(config, dict) else None
            if isinstance(workers, int):
                record(workers)


def _record_workers_from_logs(logs_dir: Path | None, record: Callable[[int], None]) -> None:
    """Inspect pipeline logs to infer Word2Vec worker counts."""

    if not logs_dir or not logs_dir.exists():
        return

    pattern = _WORD2VEC_WORKER_PATTERN
    for log_path in list(logs_dir.glob("xgb-*.err")) + list(logs_dir.glob("xgb-*.out")):
        try:
            with log_path.open("r", encoding="utf-8", errors="ignore") as handle:
                for line in handle:
                    match = pattern.search(line)
                    if not match:
                        continue
                    try:
                        record(int(match.group(1)))
                    except ValueError:  # pragma: no cover - defensive guard
                        continue
        except OSError:
            LOGGER.debug(
                "Unable to read log file %s for Word2Vec worker discovery.",
                log_path,
            )


def discover_cached_word2vec_workers(
    *, directories: Sequence[Path], logs_dir: Path | None
) -> tuple[int, ...]:
    """
    Inspect cached artefacts and logs to infer previously used Word2Vec worker counts.

    :param directories: Candidate directories that may contain vectoriser metadata.
    :type directories: Sequence[Path]
    :param logs_dir: Directory housing pipeline log files.
    :type logs_dir: Path | None
    :returns: Tuple of worker counts ordered by observed frequency (descending).
    :rtype: tuple[int, ...]
    """

    counts: Counter[int] = Counter()

    def _record(value: int) -> None:
        """Count a positive Word2Vec worker observation."""

        if value > 0:
            counts[value] += 1

    _record_workers_from_dirs(directories, _record)
    _record_workers_from_logs(logs_dir, _record)

    if not counts:
        return ()
    return tuple(value for value, _ in counts.most_common())


def apply_word2vec_workers_override(args, *, paths: PipelinePaths) -> None:
    """Override ``--word2vec-workers`` when cached artefacts suggest a value.

    This scans logs and known artefact roots under ``paths`` to infer the most
    common worker count used previously. If the current CLI value differs, it is
    overridden to maximise reuse of cached models.
    """

    logs_dir = paths.root / "logs" / "xgb"
    discovery_dirs: List[Path] = [
        paths.sweep_dir,
        paths.opinion_sweep_dir,
        paths.next_video_dir,
        paths.opinion_dir,
    ]
    if args.word2vec_model_dir:
        try:
            discovery_dirs.append(Path(args.word2vec_model_dir).resolve())
        except OSError:
            discovery_dirs.append(Path(args.word2vec_model_dir))
    unique_dirs: List[Path] = []
    seen: set[Path] = set()
    for candidate in discovery_dirs:
        try:
            exists = candidate.exists()
        except OSError:
            LOGGER.debug("Skipping Word2Vec discovery path %s due to access error.", candidate)
            continue
        if exists and candidate not in seen:
            unique_dirs.append(candidate)
            seen.add(candidate)
    workers_observed = discover_cached_word2vec_workers(
        directories=unique_dirs,
        logs_dir=logs_dir,
    )
    if workers_observed and args.word2vec_workers not in workers_observed:
        detected_workers = workers_observed[0]
        LOGGER.info(
            (
                "Detected cached Word2Vec artefacts trained with workers=%d; overriding "
                "--word2vec-workers to reuse them. Provide --word2vec-workers to force "
                "a new value."
            ),
            detected_workers,
        )
        args.word2vec_workers = detected_workers


__all__ = [
    "apply_word2vec_workers_override",
    "discover_cached_word2vec_workers",
]

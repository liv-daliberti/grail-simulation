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

"""Dataset and repository provenance helpers for the KNN evaluator."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


def collect_repo_state() -> Dict[str, Any]:
    """
    Return git provenance for the current repository head.

    :returns: Mapping containing commit hash, branch, and cleanliness metadata.
    """

    repo_root = Path(__file__).resolve().parents[3]

    def _run_git(args: Sequence[str]) -> Optional[str]:
        """
        Execute ``git`` with ``args`` and capture standard output.

        :param args: Sequence of command-line arguments passed to ``git``.
        :returns: Stripped stdout when the command succeeds, otherwise ``None``.
        """

        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo_root,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (OSError, ValueError):
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    commit = _run_git(["rev-parse", "HEAD"])
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = _run_git(["status", "--short"])
    dirty = bool(status)
    return {
        "git_commit": commit or "unknown",
        "git_branch": branch or "unknown",
        "git_dirty": dirty,
        "git_status": status or "",
    }


def dataset_split_provenance(split_ds) -> Dict[str, Any]:
    """
    Return provenance metadata for a single HF dataset split.

    :param split_ds: Dataset split object being inspected.
    :returns: Dictionary containing row counts and fingerprints when available.
    """

    provenance: Dict[str, Any] = {
        "num_rows": int(len(split_ds)),
        "fingerprint": getattr(split_ds, "_fingerprint", None),
    }
    info = getattr(split_ds, "info", None)
    revision = getattr(info, "dataset_revision", None) if info is not None else None
    if revision:
        provenance["dataset_revision"] = revision
    return provenance


def collect_dataset_provenance(dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return provenance metadata for all splits contained in ``dataset``.

    :param dataset: Dataset mapping containing the splits required for evaluation.
    :returns: Dictionary describing revisions and per-split metadata.
    """

    splits: Dict[str, Any] = {}
    revision: Optional[str] = None
    for split_name, split_ds in dataset.items():
        split_info = dataset_split_provenance(split_ds)
        splits[split_name] = split_info
        if revision is None and split_info.get("dataset_revision"):
            revision = split_info["dataset_revision"]
    return {
        "dataset_revision": revision,
        "splits": splits,
    }


__all__ = [
    "collect_dataset_provenance",
    "collect_repo_state",
    "dataset_split_provenance",
]

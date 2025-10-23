"""Dataset and study helpers shared across the KNN pipeline."""
from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Sequence, Tuple

from .opinion import DEFAULT_SPECS as _DEFAULT_OPINION_SPECS
from .pipeline_context import StudySpec

LOGGER = logging.getLogger("knn.pipeline.data")

def study_specs() -> Tuple[StudySpec, ...]:
    """
    Return the canonical participant-study specifications bundled with the project.

    :returns: Tuple of opinion-study descriptors used for sweeps and reporting.
    :rtype: Tuple[StudySpec, ...]
    """
    return tuple(StudySpec(spec.key, spec.issue, spec.label) for spec in _DEFAULT_OPINION_SPECS)

def warn_if_issue_tokens_used(args: argparse.Namespace) -> None:
    """
    Log a gentle reminder that ``--issues`` is deprecated for the pipeline.

    :param args: Parsed CLI namespace to inspect for deprecated ``--issues`` usage.
    :type args: argparse.Namespace
    :returns: None.
    :rtype: None
    """
    if not _split_tokens(getattr(args, "studies", "")) and _split_tokens(args.issues or ""):
        LOGGER.warning("`--issues` is deprecated for the pipeline; interpreting as study keys.")

def issue_slug_for_study(study: StudySpec) -> str:
    """
    Derive the filesystem slug used for artefacts associated with ``study``.

    :param study: Study specification providing issue/study slugs.
    :type study: StudySpec
    :returns: Concatenated issue and study slug (e.g. ``issue_study``).
    :rtype: str
    """
    return f"{study.issue_slug}_{study.study_slug}"

def resolve_studies(tokens: Sequence[str]) -> List[StudySpec]:
    """
    Return participant studies matching ``tokens``.

    :param tokens: Iterable of study keys or issue names received from the CLI.
    :type tokens: Sequence[str]
    :returns: Ordered list of study specifications corresponding to the provided tokens.
    :rtype: List[StudySpec]
    """
    available = list(study_specs())
    if not tokens:
        return available

    key_map = {spec.key.lower(): spec for spec in available}
    issue_map: Dict[str, List[StudySpec]] = {}
    for spec in available:
        issue_map.setdefault(spec.issue.lower(), []).append(spec)

    resolved: List[StudySpec] = []
    seen: set[str] = set()

    for token in tokens:
        normalised = token.strip().lower()
        if not normalised or normalised == "all":
            for spec in available:
                if spec.key not in seen:
                    resolved.append(spec)
                    seen.add(spec.key)
            continue
        if normalised in key_map:
            spec = key_map[normalised]
            if spec.key not in seen:
                resolved.append(spec)
                seen.add(spec.key)
            continue
        if normalised in issue_map:
            for spec in issue_map[normalised]:
                if spec.key not in seen:
                    resolved.append(spec)
                    seen.add(spec.key)
            continue
        valid = sorted({spec.key for spec in available} | {spec.issue for spec in available})
        raise ValueError(f"Unknown study token '{token}'. Expected one of {valid}.")
    return resolved

def _split_tokens(raw: str | None) -> List[str]:
    """
    Return a cleaned list of comma-separated tokens.

    :param raw: Raw comma-separated string sourced from CLI/environment.
    :type raw: str | None
    :returns: Cleaned tokens excluding whitespace-only items.
    :rtype: List[str]
    """
    if not raw:
        return []
    return [token.strip() for token in raw.split(",") if token.strip()]

__all__ = [
    "issue_slug_for_study",
    "resolve_studies",
    "study_specs",
    "warn_if_issue_tokens_used",
]

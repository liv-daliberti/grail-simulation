#!/usr/bin/env python
"""Utilities for concise logging of sparse/dense vector previews."""

from __future__ import annotations

from typing import Any, Dict, List
import logging

import numpy as np


def summarize_vector(vec: Any) -> Dict[str, object]:
    """
    Return a compact summary of a vector/matrix row for logging/debugging.

    Handles both SciPy-style sparse rows (exposing ``indices`` / ``data``) and
    dense numpy arrays or array-likes.
    """
    try:
        # Sparse path: check for .nnz presence and extract a single row preview
        nnz = int(getattr(vec, "nnz", 0))
        dim = int(vec.shape[1]) if len(getattr(vec, "shape", ())) == 2 else int(vec.shape[0])
        if nnz:
            row = vec[0] if hasattr(vec, "__getitem__") else vec
            indices = getattr(row, "indices", None)
            data = getattr(row, "data", None)
            if indices is not None and data is not None:
                preview: List[str] = [
                    f"{int(i)}:{float(v):.4f}" for i, v in zip(indices[:8], data[:8])
                ]
            else:
                preview = ["sparse"]
            return {"dim": dim, "nnz": nnz, "preview": preview}

        # Dense path: convert to flat numpy array and preview first few values.
        arr = np.asarray(vec).ravel()
        dim = int(arr.shape[0])
        preview_vals = [float(x) for x in arr[:8]]
        return {
            "dim": dim,
            "nnz": int(np.count_nonzero(arr)),
            "preview": [f"{v:.4f}" for v in preview_vals],
        }
    except Exception:  # pylint: disable=broad-except  # pragma: no cover - best effort
        # Fall back to best-effort dimension extraction.
        shape = getattr(vec, "shape", (0, 0))
        dim = int(shape[1] if len(shape) == 2 else (shape[0] if shape else 0) or 0)
        return {"dim": dim, "error": True}


def log_embedding_previews(
    vectorizer: Any,
    docs: List[str] | List[Any],
    full_vector: Any,
    *,
    logger: logging.Logger,
    tag: str,
) -> None:
    """
    Log a concise summary of base and full embeddings for the first document.

    - Extracts the base document by splitting the first entry in ``docs`` at the
      first newline (if present), then re-embeds it using ``vectorizer``.
    - Summarises both the base vector and the provided ``full_vector`` using
      :func:`summarize_vector`.
    - Emits two ``INFO`` lines using ``logger`` with the supplied ``tag``.

    Any exceptions raised during preview computation are swallowed to avoid
    disrupting training/evaluation flows.
    """

    try:
        if docs:
            sample_doc = str(docs[0])
            base_doc = sample_doc.split("\n", 1)[0]
            base_vec = vectorizer.transform([base_doc])
            base_summary = summarize_vector(base_vec)
            full_summary = summarize_vector(full_vector)
            logger.info(
                "%s base_doc dim=%s nnz=%s preview=%s",
                tag,
                base_summary.get("dim"),
                base_summary.get("nnz"),
                base_summary.get("preview"),
            )
            logger.info(
                "%s doc+tokens dim=%s nnz=%s preview=%s",
                tag,
                full_summary.get("dim"),
                full_summary.get("nnz"),
                full_summary.get("preview"),
            )
        else:
            # Fallback: only log the full vector when no docs are available.
            full_summary = summarize_vector(full_vector)
            logger.info(
                "%s dim=%s nnz=%s preview=%s",
                tag,
                full_summary.get("dim"),
                full_summary.get("nnz"),
                full_summary.get("preview"),
            )
    except Exception:  # pylint: disable=broad-except  # pragma: no cover - best effort
        pass


def log_single_embedding(vec: Any, *, logger: logging.Logger, tag: str) -> None:
    """Log a one-line summary for a single embedding vector."""

    try:
        summary = summarize_vector(vec)
        logger.info(
            "%s dim=%s nnz=%s preview=%s",
            tag,
            summary.get("dim"),
            summary.get("nnz"),
            summary.get("preview"),
        )
    except Exception:  # pylint: disable=broad-except  # pragma: no cover - best effort
        pass


__all__ = ["summarize_vector", "log_embedding_previews", "log_single_embedding"]

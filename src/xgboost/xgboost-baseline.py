#!/usr/bin/env python3
# pylint: skip-file
"""Backward-compatible entry point for the XGBoost baseline."""

from __future__ import annotations

from xgboost.cli import main


if __name__ == "__main__":  # pragma: no cover
    main()

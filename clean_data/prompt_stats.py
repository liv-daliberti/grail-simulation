#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Thin wrapper that mirrors the legacy prompt statistics entry point.

Historically the prompt reporting lived in this script; during the module
split we kept the file so downstream tooling could continue invoking it.
The implementation now simply delegates to :mod:`clean_data.prompt.cli`.
"""

from __future__ import annotations

from clean_data.prompt import generate_prompt_feature_report, main

__all__ = ["generate_prompt_feature_report", "main"]

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for prompt statistics CLI."""

from __future__ import annotations

from clean_data.prompt import generate_prompt_feature_report, main

__all__ = ["generate_prompt_feature_report", "main"]

if __name__ == "__main__":
    main()

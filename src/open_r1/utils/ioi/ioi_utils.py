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

from collections import defaultdict
from functools import lru_cache

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    load_dataset = None  # type: ignore


def add_includes(code: str, problem_id: str) -> str:
    """
    Fix common compilation errors for IOI problems.
    """
    if not code:
        return code
    # has most of the useful functions
    code_header = "#include <bits/stdc++.h>\n"
    # include the problem header
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + "\n"
    # use namespace std since models forget std:: often
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    return code_header + code


@lru_cache
def load_ioi_tests_for_year(year: int) -> dict[str, dict[str, tuple[str, str]]]:
    """
    Load IOI tests for a given year.
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' package is required to load IOI test cases. "
            "Install it with `pip install datasets`."
        )
    tests_dataset = load_dataset("open-r1/ioi-test-cases", name=f"{year}", split="train")
    test_cases = defaultdict(dict)
    for test_case in tests_dataset:
        test_cases[test_case["problem_id"]][test_case["test_name"]] = (
            test_case["test_input"],
            test_case["test_output"],
        )
    return test_cases


def load_ioi_tests(year: int, problem_id: str) -> dict[str, tuple[str, str]]:
    """
    Load IOI tests for a given year and problem id.
    """
    return load_ioi_tests_for_year(year)[problem_id]

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

"""Code execution rewards used by the GRPO training pipeline."""

from __future__ import annotations

import re
import subprocess
from typing import Any, Callable, List, Optional

__all__ = [
    "BINARY_THRESHOLD",
    "binary_code_reward",
    "code_reward",
    "extract_code",
    "get_code_format_reward",
    "match_pattern_reward",
]

BINARY_THRESHOLD = 0.99

def extract_code(completion: str, language: str = "python") -> str:
    """
    Return the last fenced code block for the requested language.

    :param completion: Completion text potentially containing code blocks.
    :type completion: str
    :param language: Code fence language to extract (defaults to ``"python"``).
    :type language: str
    :returns: Extracted code snippet or an empty string when absent.
    :rtype: str
    """
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if matches else ""
    return extracted_answer


def _run_python_test_cases(code: str, test_cases: list[dict[str, Any]], timeout: float) -> float:
    """Execute ``code`` against :mod:`python` test cases and return the success rate.

    :param code: Python source snippet to evaluate.
    :param test_cases: Iterable of dictionaries containing ``input`` and ``output`` keys.
    :param timeout: Timeout in seconds for each subprocess execution.
    :returns: Fraction of test cases that passed.
    """
    if not test_cases:
        return 0.0

    passed = 0
    for case in test_cases:
        try:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            continue

        if process.returncode != 0:
            continue

        output = process.stdout.strip()
        all_correct = True
        for actual, expected in zip(
            output.split("\n"),
            str(case["output"]).split("\n"),
        ):
            if actual.strip() != expected.strip():
                all_correct = False
                break

        if all_correct:
            passed += 1

    return passed / len(test_cases)


def code_reward(  # pylint: disable=too-many-locals
    completions,
    num_parallel: int = 2,
    enforce_same_language: bool = False,
    exec_timeout: float = 5.0,
    **kwargs,
) -> list[float]:
    """Evaluate code snippets locally using Python subprocesses.

    Assumes the dataset contains a ``verification_info`` column with test cases.

    :param completions: Chat completions containing candidate code blocks.
    :type completions: list[Any]
    :param num_parallel: Retained for backwards compatibility (unused).
    :type num_parallel: int
    :param enforce_same_language: Require all verification entries to share a language.
    :type enforce_same_language: bool
    :param exec_timeout: Timeout (seconds) for each subprocess execution.
    :type exec_timeout: float
    :param kwargs: Additional metadata, expected to contain ``verification_info``.
    :type kwargs: dict
    :returns: Success rate per completion.
    :rtype: list[float]
    """
    _ = num_parallel  # maintained for call-site compatibility

    verification_info = kwargs["verification_info"]
    if not verification_info:
        raise ValueError("verification_info is required for code_reward")

    if enforce_same_language:
        language = verification_info[0]["language"]
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("verification_info entries must share the same language")

    rewards: list[float] = []
    for completion, info in zip(completions, verification_info):
        language = info.get("language", "python")
        if language != "python":
            raise ValueError(f"Unsupported verification language: {language}")

        if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
            content = str(completion[-1].get("content", ""))
        elif isinstance(completion, dict):
            content = str(completion.get("content", ""))
        else:
            content = str(completion)

        code_snippet = extract_code(content)
        reward = _run_python_test_cases(
            code_snippet,
            info["test_cases"],
            timeout=exec_timeout,
        )
        rewards.append(reward)

    return rewards


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    enforce_same_language: bool = False,
    exec_timeout: float = 5.0,
    **kwargs,
) -> list[Optional[float]]:
    """Convert execution rewards into binary success scores.

    :param completions: Chat completions containing candidate code blocks.
    :param num_parallel: Retained for backwards compatibility (unused).
    :param enforce_same_language: Enforce a consistent language across test cases.
    :param exec_timeout: Execution timeout forwarded to the subprocess runner.
    :param kwargs: Additional metadata forwarded to :func:`code_reward`.
    :returns: Binary success indicators per completion (``None`` for missing rewards).
    """
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        enforce_same_language=enforce_same_language,
        exec_timeout=exec_timeout,
        **kwargs,
    )

    output: List[Optional[float]] = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def get_code_format_reward(language: str = "python") -> Callable[[list[Any]], list[float]]:
    """Format reward function specifically for code responses.

    :param language: Expected language for the fenced code block in the answer.
    :returns: Callable returning binary rewards based on tag and fence structure.
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        """Score completions on structural compliance with the code-format template.

        :param completions: Model completions to evaluate.
        :param kwargs: Additional keyword arguments (unused).
        :returns: Binary rewards signalling whether the format matches expectations.
        """
        _ = kwargs
        return match_pattern_reward(completions, pattern)

    return code_format_reward


def match_pattern_reward(
    completions: list[Any],
    pattern: str,
    *,
    flags: int = re.DOTALL | re.MULTILINE,
) -> list[float]:
    """Return 1.0 when the first message content matches ``pattern``.

    :param completions: Completions whose first message content is inspected.
    :param pattern: Regular expression applied to the extracted text.
    :param flags: ``re`` module flags that control matching semantics.
    :returns: Reward values where 1.0 indicates a match.
    """

    contents: list[str] = []
    for completion in completions:
        if (
            isinstance(completion, list)
            and completion
            and isinstance(completion[0], dict)
        ):
            contents.append(str(completion[0].get("content", "")))
        else:
            contents.append(str(completion))
    return [1.0 if re.match(pattern, content, flags) else 0.0 for content in contents]

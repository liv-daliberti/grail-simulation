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

import asyncio
import json
import re
from typing import Any, Callable, List, Optional

from .utils.code_providers import get_provider
from .utils.ioi import (
    SubtaskResult,
    TestResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
    score_subtask,
)

__all__ = [
    "BINARY_THRESHOLD",
    "binary_code_reward",
    "code_reward",
    "extract_code",
    "get_code_format_reward",
    "ioi_code_reward",
    "match_pattern_reward",
]

BINARY_THRESHOLD = 0.99


def _init_event_loop() -> asyncio.AbstractEventLoop:
    """Return an event loop, creating one when necessary."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(
    completions,
    test_batch_size: int = 1,
    provider_type: str = "piston",
    **kwargs,
) -> list[float]:
    """
    Evaluate IOI problems using the configured execution client.

    :param completions: Model completions containing C++ code solutions.
    :type completions: list[Any]
    :param test_batch_size: Number of tests executed per submission batch.
    :type test_batch_size: int
    :param provider_type: Execution provider to use (``\"piston\"`` or ``\"morph\"``).
    :type provider_type: str
    :param kwargs: Additional fields describing the IOI problems (e.g. ``id``).
    :type kwargs: dict
    :returns: Scores supplied by the execution client for each completion.
    :rtype: list[float]
    """

    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # For info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = []
    for completion, problem_id in zip(completions, kwargs["id"]):
        snippet = extract_code(completion[-1]["content"], "cpp")
        code_snippets.append(add_includes(snippet, problem_id))

    problems_data = [
        dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())
    ]

    loop = _init_event_loop()
    eval_tasks = [
        loop.create_task(
            score_subtask(
                execution_client,
                problem_data,
                code,
                test_batch_size=test_batch_size,
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]

    gathered = loop.run_until_complete(
        asyncio.gather(*eval_tasks, return_exceptions=True)
    )

    safe_results: List[SubtaskResult] = []
    for result in gathered:
        if isinstance(result, SubtaskResult):
            safe_results.append(result)
            continue

        if isinstance(result, BaseException):
            print(f"Error from {provider_type} worker: {result}")
            safe_results.append(SubtaskResult())
            continue

        safe_results.append(
            SubtaskResult(
                test_results=[TestResult(test_name="synthetic", score=float(result))]
            )
        )

    return [result.score for result in safe_results]


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


def code_reward(  # pylint: disable=too-many-locals
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a ``verification_info`` column with test cases.
    """
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()

            # TODO: replace with a validator that compares structured
            #       outputs rather than relying on exact stdout matches.
            all_correct = True
            for line1, line2 in zip(
                output.split('\\n'),
                case["output"].split('\\n'),
            ):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [
        extract_code(completion[-1]["content"]) for completion in completions
    ]
    verification_info = kwargs["verification_info"]

    scripts: List[str] = []
    for code, info in zip(code_snippets, verification_info):
        script = evaluation_script_template.format(
            code=json.dumps(code),
            test_cases=json.dumps(json.dumps(info["test_cases"])),
        )
        scripts.append(script)

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("verification_info entries must share the same language")

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(
        scripts,
        ["python"] * len(scripts),
    )


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[Optional[float]]:
    """Convert execution rewards into binary success scores."""
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
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
    """Format reward function specifically for code responses."""
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        _ = kwargs
        return match_pattern_reward(completions, pattern)

    return code_format_reward


def match_pattern_reward(
    completions: list[Any],
    pattern: str,
    *,
    flags: int = re.DOTALL | re.MULTILINE,
) -> list[float]:
    """Return 1.0 when the first message content matches ``pattern``."""

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

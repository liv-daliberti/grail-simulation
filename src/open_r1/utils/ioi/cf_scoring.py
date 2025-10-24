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

"""Codeforces-style scoring helpers used by the Open-R1 IOI toolkit."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Literal

from async_lru import alru_cache

try:  # pragma: no cover - optional dependency
    import pandas as pd  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pd = None  # type: ignore

try:  # pragma: no cover - optional dependency
    import aiofiles  # type: ignore
    from aiofiles import os as aiofiles_os  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    aiofiles = None  # type: ignore
    aiofiles_os = None  # type: ignore

from .piston_client import PistonClient, PistonError
from .utils import batched


async def score_single_test_case(
    client: PistonClient,
    problem_data: dict,
    *,
    test_case: tuple[str, str],
    submission: str,
    submission_language: str = "cpp",
) -> dict[str, Any] | None:
    """Score a submission against a single Codeforces-style test case.

    :param client: Piston execution client used to run the code.
    :param problem_data: Metadata describing the problem (limits, checker).
    :param test_case: ``(input, output)`` pair representing the test case.
    :param submission: Source code submitted by the model.
    :param submission_language: ``python`` or ``cpp`` selector.
    :returns: Raw execution result dictionary returned by Piston.
    :raises ValueError: If the submission language is unsupported.
    """
    test_input, test_output = test_case
    if submission_language not in ["python", "cpp"]:
        raise ValueError(f"Invalid submission language: {submission_language}")
    try:
        result = await client.send_execute(
            {
                "files": [
                    {"name": f"main.{submission_language}", "content": submission},
                    *(
                        [{"name": "checker.py", "content": problem_data["generated_checker"]}]
                        if problem_data["generated_checker"]
                        else []
                    ),
                    {"name": "input.txt", "content": test_input},
                    {"name": "correct_output.txt", "content": test_output},
                    {
                        "name": "grader_config",
                        "content": "\n".join(
                            f"{key}={value}"
                            for key, value in {
                                "TIME_LIMIT": problem_data["time_limit"],
                                "MEMORY_LIMIT": problem_data["memory_limit"],
                                "INPUT_MODE": problem_data["input_mode"],
                            }.items()
                        ),
                    },
                ],
                "run_timeout": (problem_data["time_limit"] + 10) * 1000,
                # +10 seconds hard limit. time limits are handled by the codeforces script
            },
            language="cf_python3" if submission_language == "python" else "c++17",
        )
    except (PistonError, asyncio.TimeoutError, ValueError) as exc:
        print(f"Error scoring submission: {exc}")
        return None

    return result


@alru_cache(maxsize=32)  # NOTE: cache size favors common contest reuse without exhausting memory
async def get_generated_contest_tests(contest_id: str) -> list[dict]:
    """Return Codeforces generated tests for the specified contest.

    :param contest_id: Contest identifier string.
    :returns: Mapping of problem ids to generated test-case dictionaries.
    :raises ValueError: If the tests folder is not configured or missing.
    """
    if pd is None or aiofiles is None or aiofiles_os is None:
        raise ImportError(
            "The 'pandas' and 'aiofiles' packages are required to load Codeforces generated tests. "
            "Install them with `pip install pandas aiofiles`."
        )

    tests_folder = os.environ.get("CF_TESTS_FOLDER", None)
    if not tests_folder:
        raise ValueError(
            "CF_TESTS_FOLDER environment variable not set! "
            "Download the Codeforces generated tests and set CF_TESTS_FOLDER to the folder path. "
            "See https://huggingface.co/datasets/open-r1/codeforces for more information."
        )
    if not await aiofiles_os.path.exists(tests_folder):
        raise ValueError(
            f"CF_TESTS_FOLDER path '{tests_folder}' does not exist! "
            "Download the Codeforces generated tests and update CF_TESTS_FOLDER accordingly. "
            "See https://huggingface.co/datasets/open-r1/codeforces for more information."
        )
    parquet_path = os.path.join(tests_folder, f"test_cases_{int(contest_id):04d}.parquet")
    if not await aiofiles_os.path.exists(parquet_path):
        return {}

    # Read parquet file asynchronously
    async with aiofiles.open(parquet_path, "rb") as parquet_file:
        content = await parquet_file.read()
    parquet_frame = pd.read_parquet(BytesIO(content))  # type: ignore[arg-type]

    # Group by problem_id and convert to dictionary of lists
    grouped_tests = (
        parquet_frame.groupby("problem_id")
        .apply(lambda frame: frame[["input", "output"]].to_dict("records"))
        .to_dict()
    )

    return grouped_tests


async def get_generated_tests(problem_id: str) -> list[dict]:
    """Return generated tests for a specific contest problem.

    :param problem_id: Problem identifier in ``contest/problem`` form.
    :returns: List of test case dictionaries.
    """
    contest_id = problem_id.split("/")[0]
    return (await get_generated_contest_tests(contest_id)).get(problem_id, [])


@dataclass(frozen=True)
class ScoreSubmissionConfig:
    """Configuration parameters controlling Codeforces submission scoring."""

    test_batch_size: int = 1
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum"
    no_compile_reward: float = -0.1
    no_submission_reward: float = -1.0
    submission_language: str = "cpp"


async def score_submission(
    client: PistonClient,
    problem_data: dict,
    submission: str,
    *,
    config: ScoreSubmissionConfig | None = None,
) -> float | None:  # pylint: disable=too-many-locals
    """Aggregate scores for a submission across official and generated tests.

    :param client: Piston execution client used to run batches.
    :param problem_data: Problem metadata including official tests.
    :param submission: Source code submitted by the model.
    :param config: Optional scoring configuration overrides.
    :returns: Scalar reward representing submission quality.
    :raises ValueError: If arguments specify unsupported options.
    """
    options = config or ScoreSubmissionConfig()
    if options.submission_language not in ["python", "cpp"]:
        raise ValueError(f"Invalid submission language: {options.submission_language}")
    generated_tests = await get_generated_tests(problem_data["id"])
    test_cases = problem_data["official_tests"] + generated_tests
    # invalid/not a coding problem
    if not test_cases:
        return None
    # no code extracted
    if not submission:
        return options.no_submission_reward

    passed_test_cases = 0
    # Evaluate batches sequentially; bail out as soon as a failure is observed.
    batches = (
        batched(test_cases, options.test_batch_size)
        if options.test_batch_size >= 1
        else [test_cases]
    )
    for test_batch_to_run in batches:
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client,
                        problem_data,
                        test_case=(test_case["input"], test_case["output"]),
                        submission=submission,
                        submission_language=options.submission_language,
                    )
                )
                for test_case in test_batch_to_run
            ]
        )
        if any(
            result and result["compile"]["code"] != 0
            for result in results
        ):
            return options.no_compile_reward

        tests_passed_results = [
            result
            and result["run"]["code"] == 0
            and result["run"]["stdout"].strip() == "1"
            for result in results
        ]
        if options.scoring_mode == "pass_fail" and any(
            not test_passed for test_passed in tests_passed_results
        ):
            break
        passed_test_cases += sum(
            1 for test_passed in tests_passed_results if test_passed
        )

    pass_fail_score = 1.0 if passed_test_cases == len(test_cases) else 0.0

    if options.scoring_mode == "pass_fail":
        return pass_fail_score
    if options.scoring_mode == "partial":
        return passed_test_cases / len(test_cases)
    if options.scoring_mode == "weighted_sum":
        return pass_fail_score + 0.1 * (passed_test_cases / len(test_cases))
    raise ValueError(f"Invalid scoring mode: {options.scoring_mode}")

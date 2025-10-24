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

"""IOI evaluation harness that runs submissions via the Piston sandbox."""

import asyncio
from dataclasses import asdict, dataclass, field
from typing import Optional

from .ioi_utils import load_ioi_tests
from .piston_client import PistonClient, PistonError
from .utils import batched


def _normalise_test_case(raw_case: object) -> tuple[str, Optional[str]]:
    """
    Convert a raw test-case payload into a standard ``(input, output)`` tuple.

    :param raw_case: Original representation sourced from the dataset.
    :type raw_case: object
    :returns: Two-tuple of input and optional expected output text.
    :rtype: Tuple[str, Optional[str]]
    """
    if isinstance(raw_case, (list, tuple)):
        input_part = raw_case[0] if raw_case else ""
        output_part = raw_case[1] if len(raw_case) > 1 else None
    else:
        input_part = raw_case
        output_part = None
    input_text = str(input_part) if input_part is not None else ""
    output_text = str(output_part) if output_part is not None else None
    return input_text, output_text


@dataclass
class TestResult:
    """
    Represents the result of a single test case execution.

    Attributes:
        test_name: Name of the test case
        score: Score achieved for this test (0.0 to 1.0)
        status: Status code of the test result (e.g., 'AC', 'WA', 'TLE')
        feedback: Detailed feedback message from the judge or an error message
    """

    test_name: str
    score: float = 0.0
    status: str = "SKIPPED"
    feedback: str = None


@dataclass
class SubtaskResult:
    """
    Represents the result of a subtask containing multiple test cases.

    Attributes:
        problem: Problem identifier
        subtask: Subtask identifier
        points: Maximum points available for this subtask
        score_precision: Number of decimal places for score rounding
        test_results: List of individual test case results
    """

    problem: str = None
    subtask: str = None

    points: float = 0.0
    score_precision: int = 2

    test_results: list[TestResult] = field(default_factory=list)

    @property
    def status(self):
        """
        Determines the overall status of the subtask based on the worst status among test results.
        Status priorities are ordered from worst to best.

        Returns:
            str: The status with the highest priority (lowest value)
        """
        status_prios = {
            "CE": -1,
            "RE": 0,
            "WA": 1,
            "MLE": 2,
            "TLE": 3,
            "PA": 4,
            "AC": 5,
            "SKIPPED": 999,
        }
        if not self.test_results:
            return "SKIPPED"
        return min(
            (result.status for result in self.test_results),
            key=status_prios.__getitem__,
        )

    @property
    def score(self):
        """
        Calculates the raw score for the subtask as the minimum score across all test results.

        Returns:
            float: The rounded minimum score
        """
        scores = [test_result.score for test_result in self.test_results]
        if not scores:
            return 0
        return round(min(scores), self.score_precision)

    @property
    def weighted_score(self):
        """
        Calculates the weighted score by multiplying the raw score by the available points.

        Returns:
            float: The rounded weighted score
        """
        scores = [test_result.score for test_result in self.test_results]
        if not scores:
            return 0
        return round(min(scores) * self.points, self.score_precision)

    def to_dict(self):
        """
        Converts the SubtaskResult to a dictionary representation.

        Returns:
            dict: Dictionary containing all subtask result data
        """
        return {
            "problem": self.problem,
            "subtask": self.subtask,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "points": self.points,
            "score_precision": self.score_precision,
            "status": self.status,
            "test_results": [asdict(test_result) for test_result in self.test_results],
        }


def _extract_single_status(score: float, feedback: str) -> str:
    """
    Determines the status code based on the score and feedback message.

    :param score: Numeric score between 0.0 and 1.0.
    :type score: float
    :param feedback: Feedback message from the execution.
    :type feedback: str
    :return: Status code indicating the failure mode:
        ``"CE"``, ``"MLE"``, ``"TLE"``, ``"WA"``, ``"RE"``, ``"AC"``, or ``"PA"``.
    :rtype: str
    """
    if score == 0.0:
        feedback_text = feedback or ""
        failure_map = {
            "Compilation error": "CE",
            "Memory limit exceeded": "MLE",
            "Time limit exceeded": "TLE",
            "Output isn't correct": "WA",
        }
        for hint, status in failure_map.items():
            if hint in feedback_text:
                return status
        return "RE"
    if score == 1.0:
        return "AC"
    return "PA"


async def score_single_test_case(
    client: PistonClient,
    subtask: dict,
    *,
    test_name: str,
    test_case: tuple[str, Optional[str]],
    submission: str,
) -> TestResult:
    """
    Scores a single test case by running the submission against the provided input and output.

    :param client: Piston client used to execute code.
    :type client: PistonClient
    :param subtask: Subtask configuration dictionary.
    :type subtask: dict
    :param test_name: Name of the test case.
    :type test_name: str
    :param test_input: Input data for the test case.
    :type test_input: str
    :param test_output: Expected output for the test case.
    :type test_output: str
    :param submission: Source code being evaluated.
    :type submission: str
    :return: Result of the test case execution.
    :rtype: TestResult
    """
    test_input, expected_output = test_case
    score, feedback = await run_submission(
        client,
        subtask,
        test_input,
        submission,
        expected_output,
    )
    score = float(score)

    return TestResult(
        test_name=test_name,
        score=score,
        status=_extract_single_status(score, feedback),
        feedback=feedback,
    )


async def score_subtask(
    client: PistonClient,
    subtask: dict,
    submission: str,
    test_case_run_cache: Optional[dict] = None,
    test_batch_size: int = 1,
) -> SubtaskResult:  # pylint: disable=too-many-locals
    """
    Scores all test cases in a subtask.

    :param client: Piston client instance used to execute code.
    :type client: PistonClient
    :param subtask: Subtask configuration dictionary.
    :type subtask: dict
    :param submission: Source code of the submission.
    :type submission: str
    :param test_case_run_cache: Optional cache of previously run test cases.
    :type test_case_run_cache: dict | None
    :param test_batch_size: Number of test cases to evaluate in parallel before
        checking for failures; ``-1`` evaluates all test cases concurrently.
    :type test_batch_size: int
    :return: Result of the subtask evaluation.
    :rtype: SubtaskResult
    """
    subtask_result = SubtaskResult(
        problem=subtask["id"],
        subtask=subtask["subtask"],
        points=subtask["score"],
        score_precision=subtask["score_precision"],
        test_results=[],
    )

    # tests that are not cached
    tests_to_run = [
        (ti, test_name)
        for ti, test_name in enumerate(subtask["test_names"])
        if test_case_run_cache is None or test_name not in test_case_run_cache
    ]

    # initialize test results with cached results or empty (SKIPPED) TestResult objects
    subtask_result.test_results = [
        test_case_run_cache[test_name]
        if test_case_run_cache is not None and test_name in test_case_run_cache
        else TestResult(test_name=test_name)
        for test_name in subtask["test_names"]
    ]

    # we skip submissions where no code was extracted
    # no need to do anything, as we have a failed cached result
    should_skip = not submission or any(
        test_result.status != "SKIPPED" and test_result.score == 0.0
        for test_result in subtask_result.test_results
    )
    if should_skip:
        return subtask_result

    if "test_cases" in subtask:
        test_cases = subtask["test_cases"]
        if isinstance(subtask["test_cases"], list):
            test_cases = dict(zip(subtask["test_names"], subtask["test_cases"]))
    else:
        test_cases = load_ioi_tests(subtask["year"], subtask["id"])

    # run one batch, check if any failed (0 score): stop and skip remaining tests
    for test_batch_to_run in batched(tests_to_run, test_batch_size):
        results = await asyncio.gather(
            *(
                score_single_test_case(
                    client,
                    subtask,
                    test_name=test_name,
                    test_case=_normalise_test_case(test_cases[test_name]),
                    submission=submission,
                )
                for _, test_name in test_batch_to_run
            )
        )
        for (test_index, test_name), test_result in zip(test_batch_to_run, results):
            if test_case_run_cache is not None:
                test_case_run_cache[test_name] = test_result
            subtask_result.test_results[test_index] = test_result

        # Stop early if it failed
        if any(test_result.score == 0.0 for test_result in results):
            break

    return subtask_result


async def score_subtasks(
    client: PistonClient, subtasks: list[dict], submission: str, skip_mode: bool = True
) -> list[SubtaskResult]:
    """
    Scores multiple subtasks for a submission.

    :param client: Piston client instance for executing code.
    :type client: PistonClient
    :param subtasks: Subtask configuration dictionaries.
    :type subtasks: list[dict]
    :param submission: Source code of the submission.
    :type submission: str
    :param skip_mode: If ``True``, evaluate test by test and stop after the first failure;
        otherwise run all tests in parallel. Recommended for large batches.
    :type skip_mode: bool
    :return: Results for all subtasks.
    :rtype: list[SubtaskResult]
    """
    # avoid rerunning tests present in multiple subtasks
    test_case_run_cache = {}

    results: list[SubtaskResult] = []
    for subtask in subtasks:
        result = await score_subtask(
            client,
            subtask,
            submission,
            test_case_run_cache,
            skip_mode,
        )
        results.append(result)
    return results


async def run_submission(
    client: PistonClient,
    problem: dict,
    test_input: str,
    submission: str,
    test_output: Optional[str] = None,
) -> tuple[str, str]:
    """
    Executes a submission against a test case using the Piston execution environment.

    :param client: Piston client instance for executing code.
    :type client: PistonClient
    :param problem: Problem configuration dictionary.
    :type problem: dict
    :param test_input: Input data for the test case.
    :type test_input: str
    :param submission: Source code of the submission.
    :type submission: str
    :param test_output: Optional expected output for the test case.
    :type test_output: str | None
    :return: Tuple containing ``(score, feedback)``.
    :rtype: tuple[str, str]
    """
    files = [
        {"name": f"graders/{problem['id'].lower()}.cpp", "content": submission},
        {"name": "input.txt", "content": test_input},
    ]
    if test_output:
        files.append({"name": "correct_output.txt", "content": test_output})
    files.extend(
        {"name": name, "content": content}
        for name, content in problem["grader_files"]
        if content
    )

    data = {
        "files": files,
        "run_timeout": round(
            (problem["time_limit"] + 3) * 1000
        ),  # +3 seconds hard limit. time limits are handled by the ioi script
        "run_memory_limit": problem["memory_limit"],
    }
    return await execute_ioi(client, data)


async def execute_ioi(client, data) -> tuple[str, str]:
    """
    Requests to the IOI package return the score as a float in stdout, plus
    optional feedback or errors in stderr. Returns ``(score, feedback)``.
    """
    response = await client.send_execute(data)

    if "message" in response:
        raise PistonError(response["message"])

    compile_result = response.get("compile")
    if compile_result and compile_result["code"] != 0:
        return (
            "0",
            f"Compilation error exit code {compile_result['code']}\n{compile_result['stderr']}",
        )

    if "run" not in response:
        raise PistonError(response)

    run_result = response["run"]
    if run_result["code"] == 1 and "MemoryError" in run_result["stderr"]:
        return "0", "Memory limit exceeded"

    if run_result["stdout"]:
        score, feedback = run_result["stdout"], run_result["stderr"]
    elif run_result["signal"] == "SIGKILL":
        score, feedback = "0", "Time limit exceeded"
    elif run_result["code"] != 0:
        details = ", ".join(
            [
                f"language={response['language']}",
                f"version={response['version']}",
                f"exit code={run_result['code']}",
                f"stderr={run_result['stderr']}",
                f"signal={run_result['signal']}",
            ]
        )
        raise PistonError(details)
    else:
        score, feedback = "0", "Unknown error"
    return score, feedback

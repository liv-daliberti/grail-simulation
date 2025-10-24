#!/usr/bin/env python3
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

# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

"""Reward functions used by the GRPO training pipeline."""

import asyncio
import importlib
import json
import math
import os
import re
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional, Tuple

from latex2sympy2_extended import NormalizationConfig  # pylint: disable=import-error
from math_verify import LatexExtractionConfig, parse, verify  # pylint: disable=import-error
import transformers  # pylint: disable=import-error
from transformers.utils.import_utils import _is_package_available  # pylint: disable=import-error

from .utils.code_providers import get_provider
from .utils.ioi import (
    SubtaskResult,
    TestResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
    score_subtask,
)

PURE_ACC_ENV_FLAG = "PUREACC_ALLOW_BARE_NUMBER"
PURE_ACC_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
BINARY_THRESHOLD = 0.99
LEN_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False,
            malformed_operators=False,
            basic_latex=True,
            equations=True,
            boxed=True,
            units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False,
    )
]
DEFAULT_EXTRACTION_CONFIG = [LatexExtractionConfig()]

# ── helpers ───────────────────────────────────────────────────────────

_NUM_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*[\.)]?\s*$", re.I)

def _canon(text: str) -> str:
    """Return a lowercase alphanumeric representation for fuzzy matching."""

    normalised = text.replace("’", "'").strip().lower()
    normalised = re.sub(r"\s+", " ", normalised)
    return re.sub(r"[^a-z0-9]+", "", normalised)

def _parse_slate_names(slate: str) -> Tuple[List[str], dict[int, str]]:
    """Return (names_in_order, index→name). Supports '1. Title', '1) Title', or '- Title'."""
    names: List[str] = []
    idxmap: dict[int, str] = {}
    for line in (slate or "").splitlines():
        line = line.strip()
        if not line:
            continue
        ordinal_match = re.match(r"^\s*(\d+)\s*[\.\)]\s*(.+)$", line)  # "1. Title" or "1) Title"
        if ordinal_match:
            k = int(ordinal_match.group(1))
            name = ordinal_match.group(2).strip(" -")
            names.append(name)
            idxmap[k] = name
            continue
        bullet_match = re.match(r"^\s*-\s*(.+)$", line)               # "- Title"
        if bullet_match:
            names.append(bullet_match.group(1).strip())
    if not idxmap and names:
        idxmap = {i + 1: option_name for i, option_name in enumerate(names)}
    return names, idxmap

def _gold_index_from_gold_and_slate(gold: str, slate: str) -> int:
    """Return 1-based gold index, or -1 if not resolvable."""
    gold = (gold or "").strip()
    match_num_only = _NUM_ONLY.match(gold)
    if match_num_only:
        try:
            return int(match_num_only.group(1))
        except ValueError:
            return -1
    _, idxmap = _parse_slate_names(slate)
    gcan = _canon(gold)
    if gcan:
        for key, name in idxmap.items():
            if _canon(name) == gcan:
                return key
    return -1

def _completion_text(comp: Any) -> str:
    """Extract plain assistant text from various completion shapes."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, dict):
        return str(comp.get("content", "")).strip()
    if isinstance(comp, list) and comp:
        for msg in reversed(comp):
            if isinstance(msg, dict) and "content" in msg:
                content_value = str(msg.get("content", "")).strip()
                if content_value:
                    return content_value
        try:
            return " ".join(
                str(message.get("content", "")).strip()
                for message in comp
                if isinstance(message, dict)
            )
        except (AttributeError, TypeError):
            pass
    return str(comp)

_ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
_INDEX_ONLY_RE = re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)

def _safe_int(value: Any) -> Optional[int]:
    """Return an int if possible, otherwise None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_index_from_completion(text: str, allow_bare: bool) -> Optional[int]:
    """Extract numeric answer from completion text."""
    answer_match = _ANS_PAT.search(text)
    if answer_match:
        payload = answer_match.group(1).strip()
    elif allow_bare:
        payload = text.strip()
    else:
        return None

    index_match = _INDEX_ONLY_RE.match(payload)
    if not index_match:
        return None
    return _safe_int(index_match.group(1))

def pure_accuracy_reward(  # pylint: disable=too-many-locals
    completions: List[Any],
    _answer: List[str],  # unused; kept for interface parity
    **kwargs,
) -> List[float]:
    """
    Score completions using index-only accuracy.

    Steps:

    1. Parse the numeric answer from ``<answer>…</answer>`` blocks (or bare numbers when
       the ``PUREACC_ALLOW_BARE_NUMBER`` environment flag is enabled).
    2. Compare the parsed value to ``gold_index`` (1-based).
    3. Optionally ensure ``1 <= NUMBER <= n_options`` if option counts are provided.

    Returns:
        list[float]: Per-sample accuracy scores (``1.0`` or ``0.0``).
    """
    gold_idx_arr = kwargs.get("gold_index")
    option_counts = kwargs.get("n_options")

    # Normalize to lists
    if isinstance(gold_idx_arr, int):
        gold_idx_arr = [gold_idx_arr] * len(completions)
    if gold_idx_arr is None:
        # No gold index? Nothing to score.
        return [0.0] * len(completions)

    if isinstance(option_counts, int):
        option_counts = [option_counts] * len(completions)
    if option_counts is None:
        option_counts = [None] * len(completions)

    # Optional: allow bare "3" without <think>/<answer> for early ramp
    allow_bare = (
        os.environ.get(PURE_ACC_ENV_FLAG, "0").lower() in PURE_ACC_TRUE_VALUES
    )

    outs: List[float] = []
    pred_ok = elig_ok = 0

    for comp, gidx, nopt in zip(completions, gold_idx_arr, option_counts):
        txt = _completion_text(comp)
        predicted_index = _parse_index_from_completion(txt, allow_bare)
        if predicted_index is None:
            outs.append(0.0)
            continue
        pred_ok += 1

        gold_idx = _safe_int(gidx)
        if gold_idx is None or gold_idx <= 0:
            outs.append(0.0)
            continue
        elig_ok += 1

        max_options = _safe_int(nopt)
        if max_options is not None:
            if max_options <= 0 or not 1 <= predicted_index <= max_options:
                outs.append(0.0)
                continue

        outs.append(1.0 if predicted_index == gold_idx else 0.0)

    total = len(completions)
    logger = globals().get("_wb_log")
    if callable(logger) and total:
        parse_rate = pred_ok / total
        eligible_rate = elig_ok / total
        batch_mean = sum(outs) / len(outs) if outs else math.nan
        try:
            logger(
                {
                    "reward/pure_acc/parsed_rate": parse_rate,
                    "reward/pure_acc/eligible_rate": eligible_rate,
                    "reward/pure_acc/batch_mean": batch_mean,
                }
            )
        except (TypeError, ValueError):
            pass

    return outs


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[Optional[float]]:
    """Reward function that checks if a completion matches the ground truth."""
    _ = kwargs  # Unused but kept for API compatibility
    contents = [completion[0]["content"] for completion in completions]
    rewards: list[Optional[float]] = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if gold_parsed:
            # We require the answer to be provided in correct LaTeX
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except (ValueError, TypeError) as error:
                print(
                    f"verify failed: {error}, "
                    f"answer: {answer_parsed}, gold: {gold_parsed}"
                )
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution:", sol)
        rewards.append(reward)

    return rewards


def formating(completions, **kwargs):
    """Check whether completions follow the expected think/answer tag format."""
    _ = kwargs
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [
        re.match(pattern, content, re.DOTALL | re.MULTILINE)
        for content in completion_contents
    ]
    return [1.0 if match else 0.0 for match in matches]


def format_reward(
    completions: List[List[dict]],
    solution: Optional[List[str]] = None,  # keep for backwards‐compat
    answer: Optional[List[str]] = None,  # new name
    **kwargs,
) -> List[Optional[float]]:
    """
    Return the usual format reward only if the sample's accuracy reward is 0.
    Works with either `solution=[…]` or `answer=[…]`.
    """
    golds = solution if solution is not None else answer

    acc = accuracy_reward(completions, solution=golds, **kwargs)
    fmt = formating(completions)

    gated: List[Optional[float]] = []
    for accuracy_score, format_score in zip(acc, fmt):
        if accuracy_score is None:
            gated.append(None)      # skip
        elif accuracy_score == 0.0:
            gated.append(format_score)         # wrong → formatting bonus
        else:
            gated.append(0.0)       # correct → no bonus
    return gated

def tag_count_reward(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if we produce the desired number of think and
    answer tags associated with `format_reward()`.

    Adapted from:
    https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """
    _ = kwargs

    def count_tags(text: str) -> float:
        """Return a partial reward based on the presence of required tags."""

        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(text) for text in contents]


def _compute_length_correctness(contents: List[str], solution: List[str]) -> List[bool]:
    """Return per-sample correctness for length-based rewards."""
    correctness: List[bool] = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=DEFAULT_EXTRACTION_CONFIG,
        )
        if not gold_parsed:
            correctness.append(True)
            print("Failed to parse gold solution:", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=LEN_EXTRACTION_CONFIG,
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))
    return correctness


def len_reward(
    completions: list[Dict[str, str]],
    solution: list[str],
    **kwargs,
) -> list[float]:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    :param completions: List of model completions.
    :type completions: list[Dict[str, str]]
    :param solution: List of ground truth solutions.
    :type solution: list[str]
    :param kwargs: Additional keyword arguments (unused).
    :type kwargs: dict
    :return: List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    :rtype: list[float]
    """
    _ = kwargs
    contents = [completion[0]["content"] for completion in completions]

    correctness = _compute_length_correctness(contents, solution)

    # Calculate lengths
    lengths = [len(content) for content in contents]
    min_len = min(lengths)
    max_len = max(lengths)

    # If all responses have the same length, return zero rewards
    if max_len == min_len:
        return [0.0] * len(completions)

    rewards = []
    for length, is_correct in zip(lengths, correctness):
        lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

        if is_correct:
            reward = lambda_val
        else:
            reward = min(0, lambda_val)

        rewards.append(float(reward))

    return rewards


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    """Create a cosine-scaled reward function parameterised by length."""

    def cosine_scaled_reward(completions, solution, **kwargs):  # pylint: disable=too-many-locals
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        :param completions: List of model completions.
        :type completions: list[Dict[str, str]]
        :param solution: List of ground truth solutions.
        :type solution: list[str]
        :param kwargs: Additional keyword arguments (unused).
        :type kwargs: dict
        :return: Reward list computed with the cosine schedule defined by
            ``min_value_wrong``, ``max_value_wrong``, ``min_value_correct``,
            ``max_value_correct``, and ``max_len`` captured from the outer scope.
        :rtype: list[float]
        """
        _ = kwargs
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=DEFAULT_EXTRACTION_CONFIG,
            )
            if not gold_parsed:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution:", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=LEN_EXTRACTION_CONFIG,
                extraction_mode="first_match",
            )

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward

def get_repetition_penalty_reward(
    ngram_size: int,
    max_penalty: float,
    language: str = "en",
):
    """
    Compute an n‑gram repetition penalty (Appendix C.2 of
    https://huggingface.co/papers/2502.03373).

    :param ngram_size: Size of the n-grams to inspect.
    :type ngram_size: int
    :param max_penalty: Most negative reward; must be ≤ 0.
    :type max_penalty: float
    :param language: ``"en"`` (whitespace split) or ``"zh"`` (jieba split).
    :type language: str
    :return: Function that scores completions with the configured repetition penalty.
    :rtype: Callable[[list[Any]], list[float]]
    """
    if max_penalty > 0:
        raise ValueError("max_penalty should be non‑positive")

    # ---------- tokenisers for n‑gram split ----------
    if language == "en":

        def zipngram(text: str, ngram_length: int):
            """Return iterator over n-grams and the token list for English text."""

            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_length)]), words

    elif language == "zh":
        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba for Chinese repetition reward")

        jieba_module = importlib.import_module("jieba")

        def zipngram(text: str, ngram_length: int):
            """Return iterator over n-grams and the token list for Chinese text."""

            seg_list = list(jieba_module.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_length)]), seg_list

    else:
        raise ValueError(f"Language {language!r} not supported")

    # ---------- helper: normalise each completion ----------
    def _extract_content(completion):
        """
        Normalise various completion formats into a plain string.

        :param completion: Completion represented as a string, chat message dict,
            or list of chat message dicts.
        :type completion: str | dict[str, str] | list[dict[str, str]]
        :return: Extracted completion text.
        :rtype: str
        :raises TypeError: If the completion format is unsupported.
        """
        if isinstance(completion, str):
            return completion
        if isinstance(completion, dict):  # single message dict
            return completion.get("content", "")
        if (
            isinstance(completion, list)
            and completion
            and isinstance(completion[0], dict)
        ):
            return completion[0].get("content", "")
        raise TypeError(
            f"Unsupported completion format passed to repetition reward: {type(completion)}"
        )

    # ---------- the actual reward ----------
    def repetition_penalty_reward(completions, **kwargs):
        """
        :param completions: Sequence of completions to score; accepts raw strings, a
            single completion dict, or a list of role/content dicts.
        :type completions: list[str | list[dict[str, str]] | dict[str, str]]
        :param kwargs: Additional keyword arguments (unused).
        :type kwargs: dict
        :return: One reward per completion.
        :rtype: list[float]
        """
        _ = kwargs
        rewards = []
        for raw in completions:
            text = _extract_content(raw)
            if not text:
                rewards.append(0.0)
                continue

            ngram_iter, words = zipngram(text, ngram_size)
            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            total = 0
            distinct = set()
            for ngram in ngram_iter:
                distinct.add(ngram)
                total += 1

            scaling = 1.0 - len(distinct) / total  # 0‑1, higher → more repetition
            rewards.append(scaling * max_penalty)

        return rewards

    return repetition_penalty_reward

def _init_event_loop():
    """Initialize or get the current event loop."""
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
    """Evaluate IOI problems using the configured execution client."""

    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
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
    """Return the last fenced code block for the requested language."""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if matches else ""
    return extracted_answer


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

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(  # pylint: disable=too-many-locals
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    :param completions: List of model completions to evaluate.
    :type completions: list[Any]
    :param num_parallel: Number of parallel code executions.
    :type num_parallel: int
    :param provider_type: Code execution provider to use (default ``"e2b"``).
    :type provider_type: str
    :param enforce_same_language: Whether to verify all problems share the same language.
    :type enforce_same_language: bool
    :param kwargs: Additional arguments passed to the verification layer.
    :type kwargs: dict
    :return: Reward values for each completion.
    :rtype: list[float]
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


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    :param language: Programming language supported by E2B.
    :type language: str
    :return: Reward function that checks the language-specific format.
    :rtype: Callable[[list[Any]], list[float]]
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        """Return 1.0 when the completion matches the language-specific format pattern."""

        _ = kwargs
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(pattern, content, re.DOTALL | re.MULTILINE)
            for content in completion_contents
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Penalise overlong completions without rewarding shorter ones.

    Reference: Eq. (13) from the DAPO paper
    (https://huggingface.co/papers/2503.14476)

    :param max_completion_len: Maximum allowed completion length.
    :type max_completion_len: int
    :param soft_punish_cache: Buffer length before the hard penalty applies.
    :type soft_punish_cache: int
    :return: Reward function that softly penalises long completions.
    :rtype: Callable[[list[list[int]]], list[float]]
    """

    def soft_overlong_punishment_reward(
        completion_ids: list[list[int]],
        **kwargs,
    ) -> list[float]:
        """Reward function that penalizes overlong completions."""
        _ = kwargs
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                delta = max_completion_len - soft_punish_cache - completion_length
                rewards.append(delta / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward

def get_reward_funcs(
    script_args,
    _ref_model: transformers.PreTrainedModel,
    _tokenizer: transformers.PreTrainedTokenizerBase,
) -> list[Callable]:
    """Assemble the reward functions requested by the training script."""
    enforce_same_language = getattr(script_args, "enforce_same_language", False)
    reward_funcs_registry = {
        "pure_accuracy_reward": pure_accuracy_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=enforce_same_language,
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=enforce_same_language,
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
    }

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    return reward_funcs

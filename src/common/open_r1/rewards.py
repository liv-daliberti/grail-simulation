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

"""Reward functions used by the GRPO training pipeline."""

from __future__ import annotations

import importlib
import math
import os
import re
from functools import partial, update_wrapper
from typing import Any, Callable, Dict, List, Optional, Tuple

from .pure_accuracy_utils import (
    PureAccuracyContext,
    PureAccuracyStats,
    expand_to_batch,
    log_pure_accuracy_metrics,
    parse_index_from_completion,
)

try:  # pragma: no cover - optional dependency
    from latex2sympy2_extended import NormalizationConfig  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    NormalizationConfig = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from math_verify import (  # type: ignore[import]
        LatexExtractionConfig,
        parse as _math_parse,
        verify as _math_verify,
    )
except ImportError:  # pragma: no cover - optional dependency
    LatexExtractionConfig = None  # type: ignore[assignment]

    def _math_missing(*_, **__) -> Any:
        """Raise an informative error when optional math verification deps are absent.

        :raises ImportError: Always raised to signal missing dependencies.
        :returns: Never returns; always raises ``ImportError``.
        """
        raise ImportError(
            "math_verify and latex2sympy2_extended are required for math-based rewards "
            "(pip install math-verify latex2sympy2_extended)."
        )

    parse = _math_missing  # type: ignore[assignment]
    verify = _math_missing  # type: ignore[assignment]
else:  # pragma: no cover - optional dependency
    parse = _math_parse  # type: ignore[assignment]
    verify = _math_verify  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import transformers  # type: ignore[import]
except Exception:  # pragma: no cover - optional dependency
    # Catch broad exceptions to tolerate environments with partially stubbed deps
    transformers = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    from transformers.utils.import_utils import _is_package_available  # type: ignore[attr-defined]
except (ImportError, AttributeError):  # pragma: no cover - optional dependency

    def _is_package_available(_: str) -> bool:
        """Fallback that reports transformers extras as unavailable.

        :param _: Unused package name supplied by Transformers.
        :returns: Always ``False`` to disable optional features.
        """
        return False

from .rewards_code import (
    BINARY_THRESHOLD as _BINARY_THRESHOLD,
    binary_code_reward as _binary_code_reward,
    code_reward as _code_reward,
    extract_code as _extract_code,
    get_code_format_reward as _get_code_format_reward,
    match_pattern_reward as _match_pattern_reward,
)

PURE_ACC_ENV_FLAG = "PUREACC_ALLOW_BARE_NUMBER"
PURE_ACC_TRUE_VALUES = {"1", "true", "t", "yes", "y"}
BINARY_THRESHOLD = _BINARY_THRESHOLD
binary_code_reward = _binary_code_reward
binary_code_reward.__module__ = __name__
code_reward = _code_reward
code_reward.__module__ = __name__
extract_code = _extract_code
extract_code.__module__ = __name__
get_code_format_reward = _get_code_format_reward
get_code_format_reward.__module__ = __name__
HAS_MATH_REWARD_DEPS = LatexExtractionConfig is not None and NormalizationConfig is not None
_parse_index_from_completion = parse_index_from_completion

if HAS_MATH_REWARD_DEPS:
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
else:  # pragma: no cover - optional dependency
    LEN_EXTRACTION_CONFIG = []
    DEFAULT_EXTRACTION_CONFIG = []


def _require_math_reward_deps() -> None:
    """Raise an informative error when math reward dependencies are missing.

    :returns: ``None``. Raises ``ImportError`` if optional math dependencies are absent.
    """
    if not HAS_MATH_REWARD_DEPS:
        raise ImportError(
            "math_verify and latex2sympy2_extended are required for math-based rewards. "
            "Install the optional dependencies with "
            "`pip install math-verify latex2sympy2_extended`."
        )

# ── helpers ───────────────────────────────────────────────────────────

_NUM_ONLY = re.compile(r"^\s*(?:option\s*)?(\d+)\s*[\.)]?\s*$", re.I)

def _canon(text: str) -> str:
    """Return a lowercase alphanumeric representation for fuzzy matching.

    :param text: Raw text to normalise.
    :returns: Canonical alphanumeric token.
    """

    normalised = text.replace("’", "'").strip().lower()
    normalised = re.sub(r"\s+", " ", normalised)
    return re.sub(r"[^a-z0-9]+", "", normalised)

def _parse_slate_names(slate: str) -> Tuple[List[str], dict[int, str]]:
    """Return (names_in_order, index→name) extracted from slate text.

    Supports ``\"1. Title\"``, ``\"1) Title\"``, or bullet-prefixed options.

    :param slate: Slate text containing numbered or bulleted options.
    :returns: Tuple of option names and an index-to-name mapping.
    """
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


def _validate_numeric_index(
    index: int,
    names: List[str],
    idxmap: dict[int, str],
) -> int:
    """Return a consistent 1-based index when the numeric hint is valid."""

    if index <= 0:
        return -1
    if idxmap:
        return index if index in idxmap else -1
    if names:
        return index if index <= len(names) else -1
    return -1


def _match_canonical_to_index(
    canonical_gold: str,
    names: List[str],
    idxmap: dict[int, str],
) -> int:
    """Return the index whose label matches ``canonical_gold``."""

    for idx, name in idxmap.items():
        if canonical_gold == _canon(name):
            return idx
    for position, name in enumerate(names, 1):
        if canonical_gold == _canon(name):
            return position
    return -1

def _gold_index_from_gold_and_slate(gold: str, slate: str) -> int:
    """Resolve a gold identifier to a 1-based slate index or ``-1`` when missing."""

    gold = (gold or "").strip()
    if not gold:
        return -1

    names, idxmap = _parse_slate_names(slate)

    match = _NUM_ONLY.match(gold)
    if match:
        try:
            idx = int(match.group(1))
        except ValueError:
            idx = -1
        return _validate_numeric_index(idx, names, idxmap)

    canonical = _canon(gold)
    if not canonical or not names:
        return -1

    return _match_canonical_to_index(canonical, names, idxmap)

def _completion_text(comp: Any) -> str:
    """Extract plain assistant text from various completion shapes.

    :param comp: Completion payload (string, dict, or list of messages).
    :returns: Assistant message content with whitespace trimmed.
    """
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

def pure_accuracy_reward(  # pylint: disable=too-many-locals
    completions: List[Any],
    _answer: Optional[List[str]] = None,  # unused; kept for interface parity
    **kwargs,
) -> List[float]:
    """
    Score completions on next-video accuracy and opinion-direction agreement.

    Steps:

    1. Parse the numeric answer from ``<answer>…</answer>`` (or bare numbers when
       the ``PUREACC_ALLOW_BARE_NUMBER`` environment flag is enabled) and compare it
       against ``gold_index`` (1-based).
    2. Parse ``<opinion>…</opinion>`` (or aliases) and compare it to the canonicalised
       ``opinion_direction`` label (increase/decrease/no_change).
    3. Return the average reward across the available sub-tasks so the model earns partial
       credit when only one prediction is correct.

    :param completions: Model completions whose answers should be evaluated.
    :type completions: list[Any]
    :param _answer: Optional compatibility argument (falls back to ``answer`` in ``kwargs``).
    :type _answer: list[str] | None
    :param kwargs: Additional metadata such as ``gold_index``, ``n_options``, and
        ``opinion_direction``.
    :type kwargs: dict
    :returns: Per-sample rewards in the inclusive range ``[0.0, 1.0]``.
    :rtype: list[float]
    """
    if _answer is None:
        _answer = kwargs.get("answer")
    _ = _answer  # noqa: F841  # parity with legacy signature

    total = len(completions)
    allow_bare = os.environ.get(PURE_ACC_ENV_FLAG, "0").lower() in PURE_ACC_TRUE_VALUES

    stats = PureAccuracyStats()
    context = PureAccuracyContext(
        allow_bare=allow_bare,
        stats=stats,
        completion_to_text=_completion_text,
    )
    outs: List[float] = []

    metadata_iter = zip(
        completions,
        expand_to_batch(kwargs.get("gold_index"), total),
        expand_to_batch(kwargs.get("n_options"), total),
        expand_to_batch(
            kwargs.get("opinion_direction")
            or kwargs.get("opinion_answer")
            or kwargs.get("opinion_change")
            or [],
            total,
        ),
    )
    for completion, gold_value, option_value, opinion_target in metadata_iter:
        outs.append(
            context.score(
                completion,
                gold_value,
                option_value,
                opinion_target,
            )
        )

    logger = globals().get("_wb_log")
    log_pure_accuracy_metrics(total, outs, stats, logger)
    return outs


def accuracy_reward(
    completions: list[list[dict[str, str]]],
    solution: list[str],
    **kwargs,
) -> list[Optional[float]]:
    """
    Reward function that checks if a completion matches the ground truth.

    :param completions: Chat-style completions where the first message contains
        the assistant response to evaluate.
    :type completions: list[list[dict[str, str]]]
    :param solution: Ground-truth answers expressed in LaTeX.
    :type solution: list[str]
    :param kwargs: Additional keyword arguments (unused).
    :type kwargs: dict
    :returns: Binary rewards (``1.0`` for correct, ``0.0`` for incorrect) or ``None`` when
        the example cannot be scored.
    :rtype: list[Optional[float]]
    """
    _require_math_reward_deps()
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
    """
    Check whether completions follow the expected think/answer tag format.

    :param completions: Chat-style completions produced by the policy.
    :type completions: list[list[dict[str, str]]]
    :param kwargs: Additional keyword arguments (unused).
    :type kwargs: dict
    :returns: A list indicating whether each completion matches the format.
    :rtype: list[float]
    """
    _ = kwargs
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    return _match_pattern_reward(completions, pattern)


def tag_count_reward(completions, **kwargs) -> list[float]:
    """
    Reward function that checks if we produce the desired number of think and
    answer tags associated with the formatting guard.

    Adapted from:
    https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90

    :param completions: Chat-style completions produced by the policy.
    :type completions: list[list[dict[str, str]]]
    :param kwargs: Additional keyword arguments (unused).
    :type kwargs: dict
    :returns: Partial rewards for each completion based on tag presence.
    :rtype: list[float]
    """
    _ = kwargs

    def count_tags(text: str) -> float:
        """
        Return a partial reward based on the presence of required tags.

        :param text: Completion text to inspect.
        :type text: str
        :returns: Score in ``[0, 1]`` reflecting tag compliance.
        :rtype: float
        """

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
    """Return per-sample correctness for length-based rewards.

    :param contents: Generated completion texts to evaluate.
    :param solution: Gold solutions used for verification.
    :returns: Boolean flags indicating whether each completion matches the solution.
    """
    _require_math_reward_deps()
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
    """
    Create a cosine-scaled reward function parameterised by length.

    :param min_value_wrong: Minimum reward for incorrect answers.
    :type min_value_wrong: float
    :param max_value_wrong: Maximum reward for incorrect answers.
    :type max_value_wrong: float
    :param min_value_correct: Minimum reward for correct answers.
    :type min_value_correct: float
    :param max_value_correct: Maximum reward for correct answers.
    :type max_value_correct: float
    :param max_len: Length corresponding to a full cosine cycle.
    :type max_len: int
    :returns: Configured reward function that applies cosine scaling.
    :rtype: Callable[[list[Any], list[str]], list[float]]
    """

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
        _require_math_reward_deps()
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
            """
            Return iterator over n-grams and the token list for English text.

            :param text: Text snippet to tokenise.
            :type text: str
            :param ngram_length: Length of the n-grams to extract.
            :type ngram_length: int
            :returns: Tuple containing the n-gram iterator and the token list.
            :rtype: tuple[iter, list[str]]
            """

            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_length)]), words

    elif language == "zh":
        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba for Chinese repetition reward")

        jieba_module = importlib.import_module("jieba")

        def zipngram(text: str, ngram_length: int):
            """
            Return iterator over n-grams and the token list for Chinese text.

            :param text: Text snippet to tokenise with jieba.
            :type text: str
            :param ngram_length: Length of the n-grams to extract.
            :type ngram_length: int
            :returns: Tuple containing the n-gram iterator and the token list.
            :rtype: tuple[iter, list[str]]
            """

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
        """
        Reward function that penalises overlong completions.

        :param completion_ids: Token id sequences representing generated completions.
        :type completion_ids: list[list[int]]
        :param kwargs: Additional keyword arguments (unused).
        :type kwargs: dict
        :returns: Negative rewards when completions exceed the configured limits.
        :rtype: list[float]
        """
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
    """
    Assemble the reward functions requested by the training script.

    :param script_args: Argument namespace containing reward configuration flags.
    :type script_args: argparse.Namespace
    :param _ref_model: Reference model supplied for API parity (unused).
    :type _ref_model: transformers.PreTrainedModel
    :param _tokenizer: Tokeniser supplied for API parity (unused).
    :type _tokenizer: transformers.PreTrainedTokenizerBase
    :returns: Ordered list of reward callables to evaluate during training.
    :rtype: list[Callable]
    """
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
                enforce_same_language=enforce_same_language,
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                enforce_same_language=enforce_same_language,
            ),
            binary_code_reward,
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

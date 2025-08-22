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

"""Reward functions for GRPO training."""

import asyncio
import json
import math
import re
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional, List, Any, Tuple
import os

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import transformers
from transformers import PreTrainedModel

from .utils.code_providers import get_provider
from .utils.ioi import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
    score_subtask,
)

# ── helpers ───────────────────────────────────────────────────────────

_ANSWER_TAG = re.compile(r"(?si)<think>.*?</think>.*?<answer>\s*(.*?)\s*</answer>")
_NUM_ONLY   = re.compile(r"^\s*(?:option\s*)?(\d+)\s*[\.)]?\s*$", re.I)
_FORMAT_RE  = re.compile(r"(?si)^\s*<think>.*?</think>.*?<answer>\s*\d+\s*</answer>\s*$")

def _canon(s: str) -> str:
    s = s.replace("’", "'").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)

def _parse_slate_names(slate: str) -> Tuple[List[str], dict[int, str]]:
    """Return (names_in_order, index→name). Supports '1. Title', '1) Title', or '- Title'."""
    names: List[str] = []
    idxmap: dict[int, str] = {}
    for line in (slate or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^\s*(\d+)\s*[\.\)]\s*(.+)$", line)  # "1. Title" or "1) Title"
        if m:
            k = int(m.group(1))
            name = m.group(2).strip(" -")
            names.append(name)
            idxmap[k] = name
            continue
        m = re.match(r"^\s*-\s*(.+)$", line)               # "- Title"
        if m:
            names.append(m.group(1).strip())
    if not idxmap and names:
        idxmap = {i + 1: n for i, n in enumerate(names)}
    return names, idxmap

def _gold_index_from_gold_and_slate(gold: str, slate: str) -> int:
    """Return 1-based gold index, or -1 if not resolvable."""
    gold = (gold or "").strip()
    m = _NUM_ONLY.match(gold)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return -1
    _, idxmap = _parse_slate_names(slate)
    gcan = _canon(gold)
    if gcan:
        for k, nm in idxmap.items():
            if _canon(nm) == gcan:
                return k
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
                c = str(msg.get("content", "")).strip()
                if c:
                    return c
        try:
            return " ".join(str(m.get("content", "")).strip() for m in comp if isinstance(m, dict))
        except Exception:
            pass
    return str(comp)

# ── accuracy on index, using slate_text to resolve gold name → index ───────────

def _completion_text(comp: Any) -> str:
    """
    Extract plain assistant text from a completion that may be:
      • str
      • list[dict(role, content)] (chat)
      • dict(role, content)
    Fallback to str(comp) if unknown.
    """
    if isinstance(comp, str):
        return comp
    if isinstance(comp, dict):
        return str(comp.get("content", "")).strip()
    if isinstance(comp, list) and comp:
        for msg in reversed(comp):
            if isinstance(msg, dict) and "content" in msg:
                c = str(msg.get("content", "")).strip()
                if c:
                    return c
        try:
            return " ".join(str(m.get("content","")).strip() for m in comp if isinstance(m, dict))
        except Exception:
            pass
    return str(comp)
    
_ANS_PAT = re.compile(r"(?si)<answer>\s*([^<\n]+?)\s*</answer>")
_CANON_RE = re.compile(r"[^a-z0-9]+")
_LEADING_IDX = re.compile(r'^\s*(?:option\s*)?\d+\s*[\.\):\-]\s*', re.I)

# Require digits-only answer inside <answer>
_FORMAT_RE = re.compile(r"(?si)^\s*<think>.*?</think>.*?<answer>\s*\d+\s*</answer>\s*$")
_INDEX_ONLY_RE = re.compile(r'^\s*(?:option\s*)?(\d+)\s*$', re.I)

# Keep your helpers: _ANS_PAT, _INDEX_ONLY_RE, _completion_text, etc.

def pure_accuracy_reward(
    completions: List[Any],
    answer:      List[str],   # unused here; kept for interface parity
    **kw,
) -> List[float]:
    """
    Index-only accuracy:
      1) Parse NUMBER from <answer>…</answer> (or bare number if env allows)
      2) Compare to kw['gold_index'] (1-based)
      3) Optionally check 1 <= NUMBER <= n_options
    Returns per-sample 1.0/0.0.
    """
    gold_idx_arr = kw.get("gold_index")
    n_opts_arr   = kw.get("n_options", None)

    # Normalize to lists
    if isinstance(gold_idx_arr, int):
        gold_idx_arr = [gold_idx_arr] * len(completions)
    if gold_idx_arr is None:
        # No gold index? Nothing to score.
        return [0.0] * len(completions)

    if isinstance(n_opts_arr, int):
        n_opts_arr = [n_opts_arr] * len(completions)
    if n_opts_arr is None:
        n_opts_arr = [None] * len(completions)

    # Optional: allow bare "3" without <think>/<answer> for early ramp
    ALLOW_BARE = os.environ.get("PUREACC_ALLOW_BARE_NUMBER", "0").lower() in {"1","true","t","yes","y"}

    outs: List[float] = []
    fmt_ok = pred_ok = elig_ok = 0

    for comp, gidx, nopt in zip(completions, gold_idx_arr, n_opts_arr):
        txt = _completion_text(comp)

        # Strict format gate (default): require <think>…</think><answer>NUMBER</answer>
        m = _ANS_PAT.search(txt)
        if not m and not ALLOW_BARE:
            outs.append(0.0); continue

        if m:
            payload = m.group(1).strip()
        else:
            payload = txt.strip()

        mi = None
        mm = _INDEX_ONLY_RE.match(payload)
        if mm:
            try: mi = int(mm.group(1))
            except: mi = None

        if mi is None:
            outs.append(0.0); continue
        pred_ok += 1

        try:
            gi = int(gidx)
        except:
            outs.append(0.0); continue

        if gi <= 0:
            # Unlabeled / invalid row → treat as incorrect to avoid NaNs during training
            outs.append(0.0); continue
        elig_ok += 1

        if isinstance(nopt, (int, float)) and int(nopt) > 0:
            if mi < 1 or mi > int(nopt):
                outs.append(0.0); continue

        outs.append(1.0 if mi == gi else 0.0)

    # Optional: quick logging so you can see why acc might be low
    try:
        _wb_log({
            "reward/pure_acc/parsed_rate": pred_ok / max(1, len(completions)),
            "reward/pure_acc/eligible_rate": elig_ok / max(1, len(completions)),
            "reward/pure_acc/batch_mean": float(np.mean(np.asarray(outs, dtype=np.float32))) if outs else float("nan"),
        })
    except Exception:
        pass

    return outs
    
def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
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
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards

import re


def formating(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def format_reward(
    completions: List[List[dict]],
    solution:     Optional[List[str]] = None,   # keep for backwards‐compat
    answer:       Optional[List[str]] = None,   # new name
    **kwargs,
) -> List[Optional[float]]:
    """
    Return the usual format reward only if the sample's accuracy reward is 0.
    Works with either `solution=[…]` or `answer=[…]`.
    """
    # pick whichever one you got
    golds = solution if solution is not None else answer

    acc = accuracy_reward(completions, solution=golds, **kwargs)
    fmt = formating(completions)

    gated: List[Optional[float]] = []
    for a, f in zip(acc, fmt):
        if a is None:
            gated.append(None)      # skip
        elif a == 0.0:
            gated.append(f)         # wrong → formatting bonus
        else:
            gated.append(0.0)       # correct → no bonus
    return gated

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
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
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

    Args:
        completions: List of model completions
        solution: List of ground truth solutions

    Returns:
        List of rewards where:
        - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
        - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
    """
    contents = [completion[0]["content"] for completion in completions]

    # First check correctness of answers
    correctness = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )
        if len(gold_parsed) == 0:
            # Skip unparseable examples
            correctness.append(True)  # Treat as correct to avoid penalizing
            print("Failed to parse gold solution: ", sol)
            continue

        answer_parsed = parse(
            content,
            extraction_config=[
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
            ],
            extraction_mode="first_match",
        )
        correctness.append(verify(answer_parsed, gold_parsed))

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
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
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
                ],
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

    Args:
        ngram_size: size of the n‑grams to inspect
        max_penalty: most negative reward; must be ≤ 0
        language: "en" (whitespace split) or "zh" (jieba split)
    """
    if max_penalty > 0:
        raise ValueError("max_penalty should be non‑positive")

    # ---------- tokenisers for n‑gram split ----------
    if language == "en":

        def zipngram(text: str, n: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(n)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba for Chinese repetition reward")

        import jieba  # local import keeps EN fast

        def zipngram(text: str, n: int):
            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(n)]), seg_list

    else:
        raise ValueError(f"Language {language!r} not supported")

    # ---------- helper: normalise each completion ----------
    def _extract_content(completion):
        """
        Accepts:
            • "string"  (plain completion)
            • [{"role": "...", "content": "…"}]  (chat list style)
            • {"role": "...", "content": "…"}    (single‑dict style)
        Returns:
            str
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
        Args:
            completions: list[str | list[dict] | dict]
        Returns:
            list[float]  (one reward per completion)
        """
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
            for ng in ngram_iter:
                distinct.add(ng)
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


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
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

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward

def get_reward_funcs(
        script_args,
        ref_model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
    ) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
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
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
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

    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]
    return reward_funcs
"""Unit tests targeting lightweight helpers in :mod:`open_r1.rewards`."""

from __future__ import annotations

import sys
import types

import pytest


def _install_reward_dep_stubs() -> None:
    """Provide minimal stubs for optional dependencies required to import the module."""

    if "latex2sympy2_extended" not in sys.modules:
        latex_stub = types.ModuleType("latex2sympy2_extended")

        class _NormConfig:  # noqa: D401
            """Placeholder normalisation config."""

            def __init__(self, **_kwargs):
                pass

        latex_stub.NormalizationConfig = _NormConfig
        sys.modules["latex2sympy2_extended"] = latex_stub

    if "math_verify" not in sys.modules:
        math_stub = types.ModuleType("math_verify")

        class _LatexExtractionConfig:
            def __init__(self, **_kwargs):
                pass

        def _parse(_text, **_kwargs):
            return {}

        def _verify(_gold, _answer):
            return True

        math_stub.LatexExtractionConfig = _LatexExtractionConfig
        math_stub.parse = _parse
        math_stub.verify = _verify
        sys.modules["math_verify"] = math_stub

    if "transformers" not in sys.modules:
        transformers_stub = types.ModuleType("transformers")
        utils_module = types.ModuleType("transformers.utils")
        import_utils_module = types.ModuleType("transformers.utils.import_utils")

        def _is_package_available(_name: str) -> bool:
            return False

        import_utils_module._is_package_available = _is_package_available
        utils_module.import_utils = import_utils_module
        transformers_stub.utils = utils_module

        sys.modules["transformers"] = transformers_stub
        sys.modules["transformers.utils"] = utils_module
        sys.modules["transformers.utils.import_utils"] = import_utils_module


_install_reward_dep_stubs()

from open_r1 import rewards  # pylint: disable=wrong-import-position


def test_parse_slate_names_supports_multiple_formats() -> None:
    slate = """
    1. First Choice
    2) Second Choice
    - Third Option
    """
    names, idxmap = rewards._parse_slate_names(slate)  # pylint: disable=protected-access
    assert names == ["First Choice", "Second Choice", "Third Option"]
    assert idxmap == {1: "First Choice", 2: "Second Choice", 3: "Third Option"}


@pytest.mark.parametrize(
    ("gold", "slate", "expected"),
    [
        ("2", "1. Foo\n2. Bar", 2),
        ("Option 1", "1. Foo\n2. Bar", 1),
        ("Bar", "1. Foo\n2. Bar", 2),
        ("Unknown", "1. Foo\n2. Bar", -1),
        ("", "", -1),
    ],
)
def test_gold_index_resolution_handles_numbers_and_names(
    gold: str,
    slate: str,
    expected: int,
) -> None:
    value = rewards._gold_index_from_gold_and_slate(gold, slate)  # pylint: disable=protected-access
    assert value == expected


def test_completion_text_prefers_last_non_empty_message() -> None:
    completion = [
        {"role": "system", "content": ""},
        {"role": "assistant", "content": "  "},
        {"role": "assistant", "content": "Result"},
    ]
    assert rewards._completion_text(completion) == "Result"  # pylint: disable=protected-access


def test_parse_index_from_completion_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    parser = rewards._parse_index_from_completion  # pylint: disable=protected-access

    # Without the env flag bare numbers should be ignored.
    monkeypatch.delenv(rewards.PURE_ACC_ENV_FLAG, raising=False)
    assert parser("3", allow_bare=False) is None
    assert parser("3", allow_bare=True) == 3

    # Enabling the env flag allows bare values via pure_accuracy_reward.
    monkeypatch.setenv(rewards.PURE_ACC_ENV_FLAG, "TRUE")
    assert parser("3", allow_bare=True) == 3


def test_pure_accuracy_reward_handles_varied_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    completions = [{"content": "<answer>2</answer>"}]
    rewards_outcome = rewards.pure_accuracy_reward(completions, gold_index=[2])
    assert rewards_outcome == [1.0]

    # Fails when outside the allowed options.
    rewards_outcome = rewards.pure_accuracy_reward(
        completions,
        gold_index=[2],
        n_options=[1],
    )
    assert rewards_outcome == [0.0]

    # Bare numbers only allowed when the environment flag is set.
    bare_completion = ["Just 3"]
    monkeypatch.delenv(rewards.PURE_ACC_ENV_FLAG, raising=False)
    assert rewards.pure_accuracy_reward(bare_completion, gold_index=[3]) == [0.0]

    monkeypatch.setenv(rewards.PURE_ACC_ENV_FLAG, "1")
    assert rewards.pure_accuracy_reward(bare_completion, gold_index=[3]) == [1.0]

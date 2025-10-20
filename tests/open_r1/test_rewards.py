"""Unit tests targeting lightweight helpers in :mod:`open_r1.rewards`."""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from tests.helpers.datasets_stub import ensure_datasets_stub


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

    transformers_stub = sys.modules.get("transformers")
    if transformers_stub is None:
        transformers_stub = types.ModuleType("transformers")
        sys.modules["transformers"] = transformers_stub
        utils_module = types.ModuleType("transformers.utils")
        import_utils_module = types.ModuleType("transformers.utils.import_utils")

        def _is_package_available(_name: str) -> bool:
            return False

        import_utils_module._is_package_available = _is_package_available
        utils_module.import_utils = import_utils_module

        class _PreTrainedModel:
            pass

        class _PreTrainedTokenizerBase:
            pass

        class _AutoModelForCausalLM:
            @classmethod
            def from_pretrained(cls, *args, **_kwargs):
                return cls()

        class _AutoTokenizer(_PreTrainedTokenizerBase):
            chat_template: str | None = None

            @classmethod
            def from_pretrained(cls, *args, **_kwargs):
                return cls()

        transformers_stub.PreTrainedModel = _PreTrainedModel
        transformers_stub.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
        transformers_stub.AutoModelForCausalLM = _AutoModelForCausalLM
        transformers_stub.AutoTokenizer = _AutoTokenizer
        transformers_stub.utils = utils_module
        sys.modules["transformers.utils"] = utils_module
        sys.modules["transformers.utils.import_utils"] = import_utils_module

    ensure_datasets_stub()

    trl_stub = sys.modules.get("trl")
    if trl_stub is None:
        trl_stub = types.ModuleType("trl")
        sys.modules["trl"] = trl_stub

        @dataclass
        class _BaseScriptArguments:
            pass

        @dataclass
        class _BaseTrainingConfig:
            chat_template: str | None = None
            gradient_checkpointing: bool = False

        @dataclass
        class _ModelConfig:
            model_name_or_path: str = "stub"
            model_revision: str | None = None
            trust_remote_code: bool = False
            attn_implementation: str | None = None
            torch_dtype: str | None = None

        def _get_kbit_device_map():
            return None

        def _get_quantization_config(_args):
            return None

        trl_stub.ScriptArguments = _BaseScriptArguments
        trl_stub.GRPOConfig = _BaseTrainingConfig
        trl_stub.SFTConfig = _BaseTrainingConfig
        trl_stub.ModelConfig = _ModelConfig
        trl_stub.get_kbit_device_map = _get_kbit_device_map
        trl_stub.get_quantization_config = _get_quantization_config

    if "torch" not in sys.modules:
        torch_stub = types.ModuleType("torch")

        def _getattr(name: str):
            return name

        torch_stub.float16 = "float16"
        torch_stub.float32 = "float32"
        torch_stub.__getattr__ = _getattr  # type: ignore[attr-defined]
        sys.modules["torch"] = torch_stub

    utils_stub = types.ModuleType("open_r1.utils")
    utils_stub.__path__ = []  # mark as package for submodule imports

    code_providers_stub = types.ModuleType("open_r1.utils.code_providers")

    class _DummyProvider:
        def execute_scripts(self, scripts, languages):
            return [0.0] * len(scripts)

    def _get_provider(*_args, **_kwargs):
        return _DummyProvider()

    code_providers_stub.get_provider = _get_provider

    ioi_stub = types.ModuleType("open_r1.utils.ioi")

    class _SubtaskResult:  # noqa: D401
        """Placeholder for IOI scoring result."""

    class _TestResult:
        pass

    def _add_includes(*_args, **_kwargs):
        return ""

    def _noop_client(*_args, **_kwargs):
        return None

    def _score_subtask(*_args, **_kwargs):
        return _SubtaskResult()

    ioi_stub.SubtaskResult = _SubtaskResult
    ioi_stub.TestResult = _TestResult
    ioi_stub.add_includes = _add_includes
    ioi_stub.get_morph_client_from_env = _noop_client
    ioi_stub.get_piston_client_from_env = _noop_client
    ioi_stub.score_subtask = _score_subtask

    utils_stub.code_providers = code_providers_stub
    utils_stub.ioi = ioi_stub
    utils_stub.get_dataset = lambda *args, **kwargs: None  # type: ignore[assignment]
    utils_stub.get_model = lambda *args, **kwargs: None  # type: ignore[assignment]
    utils_stub.get_tokenizer = lambda *args, **kwargs: None  # type: ignore[assignment]
    utils_stub.is_e2b_available = lambda: False  # type: ignore[assignment]
    utils_stub.is_morph_available = lambda: False  # type: ignore[assignment]

    sys.modules["open_r1.utils"] = utils_stub
    sys.modules["open_r1.utils.code_providers"] = code_providers_stub
    sys.modules["open_r1.utils.ioi"] = ioi_stub


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
    assert idxmap == {1: "First Choice", 2: "Second Choice"}


def test_parse_slate_names_without_numeric_prefixes() -> None:
    slate = "- Apple\n- Banana"
    names, idxmap = rewards._parse_slate_names(slate)  # pylint: disable=protected-access
    assert names == ["Apple", "Banana"]
    assert idxmap == {1: "Apple", 2: "Banana"}


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
    rewards_outcome = rewards.pure_accuracy_reward(completions, _answer=[""], gold_index=[2])
    assert rewards_outcome == [1.0]

    # Fails when outside the allowed options.
    rewards_outcome = rewards.pure_accuracy_reward(
        completions,
        _answer=[""],
        gold_index=[2],
        n_options=[1],
    )
    assert rewards_outcome == [0.0]

    # Bare numbers only allowed when the environment flag is set.
    bare_completion = ["3"]
    monkeypatch.delenv(rewards.PURE_ACC_ENV_FLAG, raising=False)
    assert rewards.pure_accuracy_reward(bare_completion, _answer=[""], gold_index=[3]) == [0.0]

    monkeypatch.setenv(rewards.PURE_ACC_ENV_FLAG, "1")
    assert rewards.pure_accuracy_reward(bare_completion, _answer=[""], gold_index=[3]) == [1.0]

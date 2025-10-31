# noqa: D100
from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple, Type


def _make_module(name: str, attrs: Dict[str, Any]) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _ensure_package(name: str, search_path: Iterable[str] | None = None) -> types.ModuleType:
    module = _make_module(name, {})
    module.__path__ = list(search_path or [])
    module.__package__ = name
    module.__spec__ = types.SimpleNamespace(submodule_search_locations=module.__path__)
    return module


def _stub_common_modules(monkeypatch: Any) -> None:
    common_pkg = _ensure_package("common")
    monkeypatch.setitem(sys.modules, "common", common_pkg)

    open_r1_pkg = _ensure_package("common.open_r1")
    monkeypatch.setitem(sys.modules, "common.open_r1", open_r1_pkg)

    configs_mod = _make_module(
        "common.open_r1.configs",
        {
            "GRPOConfig": type("GRPOConfig", (), {}),
            "GRPOScriptArguments": type("GRPOScriptArguments", (), {}),
        },
    )
    monkeypatch.setitem(sys.modules, "common.open_r1.configs", configs_mod)

    parse_calls: list[Tuple[Callable[..., Any], Tuple[Any, ...]]] = []

    def _record_parse_and_run(main_fn: Callable[..., Any], arg_defs: Tuple[Type[Any], ...]) -> None:
        parse_calls.append((main_fn, arg_defs))

    shared_mod = _make_module(
        "common.open_r1.shared",
        {
            "execute_grpo_pipeline": lambda *args, **kwargs: None,
            "make_grpo_execute_kwargs": lambda *args, **kwargs: {},
            "parse_and_run": _record_parse_and_run,
            "build_default_component_factory": lambda: {},
            "_parse_calls": parse_calls,
        },
    )
    monkeypatch.setitem(sys.modules, "common.open_r1.shared", shared_mod)


def _stub_grail_modules(monkeypatch: Any, script_dir: Path) -> None:
    grail_pkg = _ensure_package("grail", [str(script_dir)])
    monkeypatch.setitem(sys.modules, "grail", grail_pkg)

    grail_dataset_mod = _make_module(
        "grail.grail_dataset",
        {
            "PASSTHROUGH_FIELDS": (),
            "TRAIN_KEEP_COLUMNS": (),
            "_build_dataset_and_tokenizer": lambda *args, **kwargs: ({"train": []}, object()),
            "_grail_extra_fields": lambda *args, **kwargs: {},
            "_prepare_dataset": lambda *args, **kwargs: {},
        },
    )
    monkeypatch.setitem(sys.modules, "grail.grail_dataset", grail_dataset_mod)

    grail_gail_mod = _make_module(
        "grail.grail_gail",
        {
            "OnlineDiscriminator": type("OnlineDiscriminator", (), {}),
            "RewardContext": type("RewardContext", (), {}),
            "_build_reward_contexts": lambda *args, **kwargs: [],
            "_context_from_completion": lambda *args, **kwargs: {},
            "_render_disc_text": lambda *args, **kwargs: "",
            "_select_disc_device": lambda *args, **kwargs: "cpu",
            "_train_discriminator_from_contexts": lambda *args, **kwargs: None,
            "make_gail_reward_fn": lambda *args, **kwargs: (lambda _: 0.0),
        },
    )
    monkeypatch.setitem(sys.modules, "grail.grail_gail", grail_gail_mod)

    grail_mixer_mod = _make_module(
        "grail.grail_mixer",
        {
            "LearnableRewardMixer": type("LearnableRewardMixer", (), {}),
            "MixerSetup": type("MixerSetup", (), {}),
        },
    )
    monkeypatch.setitem(sys.modules, "grail.grail_mixer", grail_mixer_mod)

    grail_rewards_mod = _make_module(
        "grail.grail_rewards",
        {
            "_adjust_reward_weights": lambda *args, **kwargs: None,
            "_apply_reward_mixer": lambda *args, **kwargs: [],
            "_maybe_enable_gail": lambda *args, **kwargs: False,
            "_resolve_reward_functions": lambda *args, **kwargs: [],
        },
    )
    monkeypatch.setitem(sys.modules, "grail.grail_rewards", grail_rewards_mod)

    grail_torch_mod = _make_module(
        "grail.grail_torch",
        {
            "nn": object(),
            "optim": object(),
            "torch": object(),
            "resolve_torch_modules": lambda: None,
        },
    )
    monkeypatch.setitem(sys.modules, "grail.grail_torch", grail_torch_mod)

    grail_utils_mod = _make_module(
        "grail.grail_utils",
        {
            "_completion_text": lambda *args, **kwargs: "",
            "_ensure_list": lambda value: list(value) if isinstance(value, (list, tuple)) else [value],
            "_parse_index_from_answer_block": lambda *args, **kwargs: 0,
            "_safe_int": lambda value, default=0: default if value is None else int(value),
        },
    )
    monkeypatch.setitem(sys.modules, "grail.grail_utils", grail_utils_mod)


def _stub_external_deps(monkeypatch: Any) -> None:
    transformers_mod = _make_module("transformers", {"set_seed": lambda *_args, **_kwargs: None})
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    trl_mod = _make_module("trl", {"ModelConfig": type("ModelConfig", (), {})})
    monkeypatch.setitem(sys.modules, "trl", trl_mod)


def test_grail_entrypoint_initialises_when_invoked_as_script(monkeypatch: Any) -> None:
    for prefix in ("grail", "common", "transformers", "trl"):
        targets = [name for name in sys.modules if name == prefix or name.startswith(f"{prefix}.")]
        for target in targets:
            monkeypatch.delitem(sys.modules, target, raising=False)

    script_path = Path(__file__).resolve().parents[2] / "src" / "grail" / "grail.py"

    _stub_external_deps(monkeypatch)
    _stub_common_modules(monkeypatch)
    _stub_grail_modules(monkeypatch, script_path.parent)

    globals_ns = runpy.run_path(str(script_path), run_name="__main__")

    shared_mod = sys.modules["common.open_r1.shared"]
    parse_calls = getattr(shared_mod, "_parse_calls")

    assert parse_calls, "parse_and_run should be invoked when grail.py runs as __main__"
    main_fn, arg_defs = parse_calls[-1]
    assert callable(main_fn)
    assert globals_ns["COMPONENT_FACTORY"] == {}, "stub factory expected to produce empty dict"
    assert arg_defs == (sys.modules["common.open_r1.configs"].GRPOScriptArguments,
                        sys.modules["common.open_r1.configs"].GRPOConfig,
                        sys.modules["trl"].ModelConfig)

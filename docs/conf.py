"""Sphinx configuration for the GRAIL Simulation project."""

from __future__ import annotations

import contextlib
import os
import sys
import types
from datetime import datetime

# Make the project root and src directory importable so autodoc can find modules.
ROOT_DIR = os.path.abspath("..")
SRC_DIR = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, ROOT_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


def _ensure_trl_stub() -> None:
    """Provide a minimal ``trl`` stub so autodoc can import open_r1 modules."""
    if os.environ.get("SPHINX_USE_REAL_TRL"):
        return

    sys.modules.pop("trl", None)
    trl_module = types.ModuleType("trl")

    class _Stub:
        """Placeholder object used for optional TRL types."""

        def __init__(self, *args, **kwargs):
            pass

    def _make_stub_class(name: str, namespace: dict[str, object] | None = None):
        namespace = namespace or {}
        namespace.setdefault("__module__", "trl")
        return type(name, (_Stub,), namespace)

    class GRPOTrainer(_Stub):
        __module__ = "trl"

        def add_callback(self, *args, **kwargs):
            return None

    trl_module.GRPOTrainer = GRPOTrainer
    trl_module.ScriptArguments = _make_stub_class("ScriptArguments")
    trl_module.GRPOConfig = _make_stub_class("GRPOConfig")
    trl_module.SFTConfig = _make_stub_class("SFTConfig")
    trl_module.ModelConfig = _make_stub_class("ModelConfig")

    class TrlParser(_Stub):
        __module__ = "trl"

        @staticmethod
        def parse_args(*_args, **_kwargs):
            return {}

    trl_module.TrlParser = TrlParser

    def _noop(*_args, **_kwargs):
        return None

    trl_module.get_peft_config = _noop
    trl_module.get_kbit_device_map = _noop
    trl_module.get_quantization_config = _noop

    extras_module = types.ModuleType("trl.extras")
    profiling_module = types.ModuleType("trl.extras.profiling")
    profiling_module.profiling_context = contextlib.nullcontext
    extras_module.profiling = profiling_module

    trainer_module = types.ModuleType("trl.trainer")
    grpo_trainer_module = types.ModuleType("trl.trainer.grpo_trainer")
    grpo_trainer_module.unwrap_model_for_generation = _noop
    grpo_trainer_module.pad = _noop
    grpo_trainer_module.GRPOTrainer = GRPOTrainer
    trainer_module.grpo_trainer = grpo_trainer_module

    data_utils_module = types.ModuleType("trl.data_utils")
    data_utils_module.maybe_apply_chat_template = _noop
    data_utils_module.is_conversational = lambda *_args, **_kwargs: False

    sys.modules["trl"] = trl_module
    sys.modules["trl.extras"] = extras_module
    sys.modules["trl.extras.profiling"] = profiling_module
    sys.modules["trl.trainer"] = trainer_module
    sys.modules["trl.trainer.grpo_trainer"] = grpo_trainer_module
    sys.modules["trl.data_utils"] = data_utils_module


_ensure_trl_stub()

project = "GRAIL Simulation"
author = "GRAIL Simulation Team"
year = datetime.utcnow().year
copyright = f"{year}, {author}"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_mock_imports = [
    "accelerate",
    "aiofiles",
    "async_lru",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "distilabel",
    "e2b_code_interpreter",
    "einops",
    "graphviz",
    "hf_transfer",
    "huggingface_hub",
    "jieba",
    "langdetect",
    "latex2sympy2_extended",
    "liger_kernel",
    "lighteval",
    "math_verify",
    "matplotlib",
    "morphcloud",
    "numpy",
    "openai",
    "pandas",
    "peft",
    "pyarrow",
    "pydantic",
    "safetensors",
    "scipy",
    "scikit_learn",
    "sklearn",
    "sentencepiece",
    "torch",
    "transformers",
    "vllm",
    "wandb",
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_typehints = "description"
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
html_static_path = ["_static"]

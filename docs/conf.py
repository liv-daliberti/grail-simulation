"""Sphinx configuration for the GRAIL Simulation project."""

from __future__ import annotations

import contextlib
import importlib.util
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
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

myst_heading_anchors = 3

autodoc_mock_imports = [
    "accelerate",
    "aiofiles",
    "async_lru",
    "bitsandbytes",
    "datasets",
    "deepspeed",
    "distilabel",
    "dotenv",
    "e2b_code_interpreter",
    "einops",
    "graphviz",
    "hf_transfer",
    "huggingface_hub",
    "jieba",
    "langdetect",
    "gensim",
    "gensim.models",
    "gensim.models.word2vec",
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
    "sentence_transformers",
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
autodoc_type_aliases = {
    "Future": "concurrent.futures.Future",
}
napoleon_include_init_with_doc = True

templates_path = ["_templates"]
exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_static_path = ["_static"]
if (
    os.environ.get("SPHINX_ENABLE_FURO", "").lower() in {"1", "true", "yes"}
    or importlib.util.find_spec("furo") is not None
):
    html_theme = "furo"
    html_theme_options = {
        "light_css_variables": {
            "color-brand-primary": "#1f4b8e",
            "color-brand-content": "#1f4b8e",
        },
        "dark_css_variables": {
            "color-brand-primary": "#8bbcef",
            "color-brand-content": "#8bbcef",
        },
    }
else:
    html_theme = "alabaster"
    html_theme_options = {}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "transformers": ("https://huggingface.co/docs/transformers/main/en", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "graphviz": ("https://graphviz.readthedocs.io/en/stable/", None),
    "datasets": ("https://huggingface.co/docs/datasets/main/en", None),
}

if os.environ.get("SPHINX_ENABLE_INTERSPHINX", "").lower() not in {"1", "true", "yes"}:
    # Avoid noisy warnings when building docs in restricted or offline environments.
    intersphinx_mapping = {}

_nitpick_targets = {
    "py:class": [
        "DatasetDict",
        "Execution",
        "ModelConfig",
        "Path",
        "argparse.ArgumentParser",
        "argparse.Namespace",
        "abc.ABC",
        "collections.Counter",
        "common.title_index.TitleResolver",
        "concurrent.futures._base.Future",
        "concurrent.futures.Future",
        "datasets.Dataset",
        "datasets.DatasetDict",
        "distilabel.pipeline.Pipeline",
        "graphviz.Digraph",
        "logging.Logger",
        "knn.index.SlateQueryConfig",
        "numpy.ndarray",
        "openai.AzureOpenAI",
        "optional",
        "pandas.DataFrame",
        "pandas.Series",
        "pd.DataFrame",
        "pathlib.Path",
        "sequence-like",
        "sklearn.feature_extraction.text.TfidfVectorizer",
        "sklearn.preprocessing.LabelEncoder",
        "torch.device",
        "torch.utils.data.Dataset",
        "transformers.AutoModelForCausalLM",
        "transformers.PreTrainedModel",
        "transformers.PreTrainedTokenizer",
        "transformers.PreTrainedTokenizerBase",
        "transformers.TrainerCallback",
        "transformers.TrainerControl",
        "transformers.TrainerState",
        "transformers.TrainingArguments",
        "transformers.generation.utils.GenerationMixin",
        "trl.GRPOConfig",
        "trl.ModelConfig",
        "trl.SFTConfig",
        "trl.ScriptArguments",
        "trl._ensure_trl_stub.<locals>.GRPOTrainer",
        "xgboost.XGBClassifier",
        "rewards where",
        "Word2VecConfig",
        "Future",
    ],
    "py:func": [
        "load_participant_allowlists",
    ],
    "py:mod": [
        "xgboost",
    ],
    "py:data": [
        "sys.argv",
    ],
}

nitpick_ignore: list[tuple[str, str]] = []
for domain, targets in _nitpick_targets.items():
    nitpick_ignore.extend((domain, target) for target in targets)

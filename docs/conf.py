"""Sphinx configuration for the GRAIL Simulation project."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the project root and src directory importable so autodoc can find modules.
ROOT_DIR = os.path.abspath("..")
sys.path.insert(0, os.path.abspath(".."))

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
    "knn",
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
    "open_r1",
    "pandas",
    "peft",
    "prompt_builder",
    "pyarrow",
    "safetensors",
    "visualization",
    "scikit_learn",
    "sklearn",
    "sentencepiece",
    "torch",
    "transformers",
    "trl",
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

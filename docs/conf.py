"""Sphinx configuration for the GRAIL Simulation project."""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Make the project root and src directory importable so autodoc can find modules.
ROOT_DIR = os.path.abspath("..")
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, ROOT_DIR)
sys.path.insert(0, SRC_DIR)

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

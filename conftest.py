"""Pytest configuration for repository-wide test helpers.

Installs or repairs a minimal torch stub before each test. Some tests inject
an ultra-minimal ``torch`` stub whose ``__getattr__`` returns strings; that
breaks optional integrations (e.g. Hugging Face datasets' dill pickler) which
expect ``torch.Tensor`` and ``torch.nn.Module`` to be real classes. This fixture
ensures the required attributes exist and are class-like.
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

# Ensure unit tests resolve user-scoped paths relative to the repository PWD.
# This avoids writes into a real user $HOME and keeps all runtime artefacts
# confined to the working tree during test runs.
_PWD = Path.cwd()
_CACHE = _PWD / ".cache"
_CONFIG = _PWD / ".config"
_DATA = _PWD / ".local" / "share"
_TMP = _PWD / ".tmp"
_PYC = _CACHE / "pyc"
for _d in (_CACHE, _CONFIG, _DATA, _TMP, _PYC, _CACHE / "huggingface"):
    _d.mkdir(parents=True, exist_ok=True)

# Point HOME and common cache/config locations at PWD-scoped directories
os.environ["HOME"] = str(_PWD)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE))
os.environ.setdefault("XDG_CONFIG_HOME", str(_CONFIG))
os.environ.setdefault("XDG_DATA_HOME", str(_DATA))
os.environ.setdefault("HF_HOME", str(_CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(_CACHE / "huggingface"))
os.environ.setdefault("PYTORCH_HOME", str(_CACHE / "torch"))
os.environ.setdefault("PIP_CACHE_DIR", str(_CACHE / "pip"))
os.environ.setdefault("TMPDIR", str(_TMP))
os.environ.setdefault("PYTHONPYCACHEPREFIX", str(_PYC))

# Ensure deprecated Transformers env is not present to avoid warnings
os.environ.pop("TRANSFORMERS_CACHE", None)


@pytest.fixture(autouse=True)
def _ensure_torch_and_transformers_stubs():
    """Guarantee minimal torch/transformers stubs are present for every test.

    Tests only need a handful of class-like attributes so that optional
    integrations (HF datasets + dill pickler) can register custom reducers
    without importing heavy dependencies.
    """
    import sys
    import types

    try:
        import torch  # type: ignore
    except ModuleNotFoundError:
        # Install a minimal stub if torch is unavailable
        module = types.ModuleType("torch")
        module.Tensor = type("Tensor", (), {})
        module.Generator = type("Generator", (), {})
        module.tensor = lambda data, dtype=None, device=None: data  # noqa: ARG005
        module.float32 = "float32"
        module.float64 = "float64"
        module.cuda = types.SimpleNamespace(is_available=lambda: False)
        nn_mod = types.ModuleType("torch.nn")
        nn_mod.Module = type("Module", (), {})
        module.nn = nn_mod  # type: ignore[attr-defined]
        dist_mod = types.ModuleType("torch.distributed")
        dist_mod.is_available = lambda: False  # type: ignore[attr-defined]
        module.distributed = dist_mod  # type: ignore[attr-defined]
        sys.modules["torch"] = module
        sys.modules["torch.nn"] = nn_mod
        sys.modules["torch.distributed"] = dist_mod
    else:
        # Repair incomplete stubs where __getattr__ returns strings
        if not isinstance(getattr(torch, "Tensor", type), type):
            setattr(torch, "Tensor", type("Tensor", (), {}))
        if not hasattr(torch, "Generator") or not isinstance(
            getattr(torch, "Generator"), type
        ):
            setattr(torch, "Generator", type("Generator", (), {}))
        nn = getattr(torch, "nn", None)
        if not isinstance(nn, types.ModuleType) or not hasattr(nn, "Module") or not isinstance(
            getattr(nn, "Module", object), type
        ):
            nn_mod = types.ModuleType("torch.nn")
            nn_mod.Module = type("Module", (), {})
            torch.nn = nn_mod  # type: ignore[attr-defined]
            sys.modules["torch.nn"] = nn_mod
        dist = getattr(torch, "distributed", None)
        if not isinstance(dist, types.ModuleType) or not hasattr(dist, "is_available"):
            dist_mod = types.ModuleType("torch.distributed")
            dist_mod.is_available = lambda: False  # type: ignore[attr-defined]
            torch.distributed = dist_mod  # type: ignore[attr-defined]
            sys.modules["torch.distributed"] = dist_mod

    # Provide a minimal transformers stub so HF datasets' dill helpers can
    # register picklers without importing the real library.
    try:
        import transformers  # type: ignore
    except ModuleNotFoundError:
        tmod = types.ModuleType("transformers")
        tmod.PreTrainedTokenizerBase = type("PreTrainedTokenizerBase", (), {})
        sys.modules["transformers"] = tmod
    else:
        if not hasattr(transformers, "PreTrainedTokenizerBase") or not isinstance(
            getattr(transformers, "PreTrainedTokenizerBase"), type
        ):
            setattr(
                transformers,
                "PreTrainedTokenizerBase",
                type("PreTrainedTokenizerBase", (), {}),
            )
    yield

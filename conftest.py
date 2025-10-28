"""Pytest configuration for repository-wide test helpers.

Installs or repairs a minimal torch stub before each test. Some tests inject
an ultra-minimal ``torch`` stub whose ``__getattr__`` returns strings; that
breaks optional integrations (e.g. Hugging Face datasets' dill pickler) which
expect ``torch.Tensor`` and ``torch.nn.Module`` to be real classes. This fixture
ensures the required attributes exist and are class-like.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _ensure_torch_stub():
    """Guarantee a compatible torch stub is present for every test."""
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
    yield

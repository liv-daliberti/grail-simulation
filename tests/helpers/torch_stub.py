"""Minimal torch stub for environments without PyTorch."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace


def _install_stub() -> ModuleType:
    module = ModuleType("torch")

    class Tensor:  # pylint: disable=too-few-public-methods
        """Placeholder tensor type used only for isinstance checks."""

    class Generator:  # pylint: disable=too-few-public-methods
        """Placeholder random generator type."""

    def tensor(data, dtype=None, device=None):  # noqa: D401, ARG002
        """Return input data unchanged; placeholder for torch.tensor."""
        return data

    module.Tensor = Tensor
    module.Generator = Generator
    module.tensor = tensor
    module.float32 = "float32"
    module.float64 = "float64"
    module.device = lambda name=None: (name or "cpu")  # noqa: ARG005
    module.cuda = SimpleNamespace(is_available=lambda: False)
    module.__dict__["__version__"] = "0.0.0"

    dist = ModuleType("torch.distributed")
    dist.is_available = lambda: False
    module.distributed = dist

    nn = ModuleType("torch.nn")

    class Module:  # pylint: disable=too-few-public-methods
        """Placeholder base class mimicking torch.nn.Module."""

    nn.Module = Module
    module.nn = nn

    return module


def ensure_torch_stub() -> None:
    """Install a lightweight torch stub when PyTorch is unavailable."""

    try:  # pragma: no cover - executed only with real torch
        import torch  # type: ignore
    except ModuleNotFoundError:
        stub = _install_stub()
        sys.modules["torch"] = stub
        sys.modules.setdefault("torch.distributed", stub.distributed)
        sys.modules.setdefault("torch.nn", stub.nn)
        return

    if not isinstance(getattr(torch, "Tensor", type), type):
        setattr(torch, "Tensor", type("Tensor", (), {}))
    if not hasattr(torch, "Generator"):
        setattr(torch, "Generator", type("Generator", (), {}))
    if not hasattr(torch, "nn"):
        nn = ModuleType("torch.nn")
        nn.Module = type("Module", (), {})
        torch.nn = nn  # type: ignore[attr-defined]

    torch.distributed = getattr(torch, "distributed", ModuleType("torch.distributed"))
    torch.distributed.is_available = getattr(torch.distributed, "is_available", lambda: False)
    sys.modules.setdefault("torch.distributed", torch.distributed)
    sys.modules.setdefault("torch.nn", torch.nn)

"""Torch dependency management helpers for the GRAIL pipeline."""

# pylint: disable=invalid-name

from __future__ import annotations

import inspect
from typing import Tuple

try:  # pragma: no cover - optional dependency
    import torch  # pylint: disable=import-error
    from torch import nn, optim  # pylint: disable=import-error
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    nn = None  # type: ignore[assignment]  # pylint: disable=invalid-name
    optim = None  # type: ignore[assignment]  # pylint: disable=invalid-name

from common.open_r1.torch_stub_utils import build_torch_stubs


def _validate_torch_modules(module, module_nn, module_optim) -> bool:
    """Return whether the provided torch modules expose the expected API surface.

    :param module: Imported ``torch`` package or stub replacement.
    :param module_nn: Module providing neural-network utilities (``torch.nn``).
    :param module_optim: Module providing optimisers (``torch.optim``).
    :returns: ``True`` when the supplied modules satisfy the expected interface.
    """

    try:
        if not inspect.isclass(getattr(module_nn, "Module", None)):  # type: ignore[arg-type]
            raise AttributeError("nn.Module is not a class")
        if not callable(getattr(module_optim, "Adam", None)):  # type: ignore[arg-type]
            raise AttributeError("optim.Adam is not callable")
        if not hasattr(module, "cuda") or not hasattr(module.cuda, "is_available"):
            raise AttributeError("torch.cuda missing expected attributes")
    except AttributeError:
        return False
    return True


def resolve_torch_modules() -> Tuple[object, object, object]:
    """Return ``torch``, ``nn``, and ``optim`` objects with stub fallbacks as needed.

    :returns: Tuple containing ``torch``, ``nn`` and ``optim`` modules.
    """

    global torch, nn, optim  # pylint: disable=global-statement

    if torch is None or nn is None or optim is None:
        torch, nn, optim = build_torch_stubs()
    elif not _validate_torch_modules(torch, nn, optim):
        torch, nn, optim = build_torch_stubs()

    return torch, nn, optim


torch, nn, optim = resolve_torch_modules()  # resolved at import time for consumers

__all__ = ["torch", "nn", "optim", "resolve_torch_modules"]

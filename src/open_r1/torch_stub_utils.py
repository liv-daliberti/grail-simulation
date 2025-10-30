"""Utilities for building lightweight PyTorch fallbacks during documentation."""

from __future__ import annotations

import types
from typing import Any, Dict, List, Optional, Sequence, Tuple

TORCH_FALLBACK_DEVICE = "cpu"


class TensorStub:
    """Minimal tensor stub to satisfy documentation builds without torch."""

    def __init__(
        self,
        data: Optional[Sequence[float]] = None,
        *,
        device: Any = None,
        **_unused: Any,
    ):
        """Initialise the stub tensor.

        :param data: Optional backing sequence that mimics tensor storage.
        :param device: Device identifier associated with the stub tensor.
        :param _unused: Extra keyword arguments accepted for API parity.
        """
        self._data = list(data) if data is not None else []
        self.device = device or TORCH_FALLBACK_DEVICE

    def numel(self) -> int:
        """Return the simulated number of elements.

        :returns: Count of pseudo elements held by the stub tensor.
        """
        return len(self._data)

    def sum(self, dim: int | None = None):
        """Return ``self`` to imitate ``torch.Tensor.sum``.

        :param dim: Unused dimension argument kept for signature parity.
        :returns: The current stub tensor instance.
        """
        del dim
        return self

    def unsqueeze(self, dim: int):
        """Return ``self`` to mirror ``torch.Tensor.unsqueeze``.

        :param dim: Dimension index that is ignored by the stub.
        :returns: The current stub tensor instance.
        """
        del dim
        return self

    def detach(self):
        """Return ``self`` because gradients are not tracked.

        :returns: The current stub tensor instance.
        """
        return self

    def cpu(self):
        """Return ``self`` since the stub already behaves as a CPU tensor.

        :returns: The current stub tensor instance.
        """
        return self

    def mean(self):
        """Return ``self`` to emulate ``torch.Tensor.mean``.

        :returns: The current stub tensor instance.
        """
        return self

    def item(self) -> float:
        """Return the first stored value or ``0.0`` when empty.

        :returns: Scalar float representation of the stub tensor.
        """
        return float(self._data[0]) if self._data else 0.0

    def to(self, *_args, **_kwargs):
        """Return ``self`` regardless of target dtype or device.

        :param _args: Positional arguments accepted for compatibility.
        :param _kwargs: Keyword arguments accepted for compatibility.
        :returns: The current stub tensor instance.
        """
        return self

    def tolist(self) -> List[float]:
        """Return a list copy of the backing data.

        :returns: List representation of the stub tensor contents.
        """
        return list(self._data)

    def __mul__(self, _other):
        """Return ``self`` for compatibility with scalar multiplication.

        :param _other: Ignored operand kept for signature parity.
        :returns: The current stub tensor instance.
        """
        return self

    def __rmul__(self, _other):
        """Return ``self`` when multiplied from the right.

        :param _other: Ignored operand kept for signature parity.
        :returns: The current stub tensor instance.
        """
        return self

    def __add__(self, _other):
        """Return ``self`` for addition operations.

        :param _other: Ignored operand kept for signature parity.
        :returns: The current stub tensor instance.
        """
        return self

    def __radd__(self, _other):
        """Return ``self`` when added from the right.

        :param _other: Ignored operand kept for signature parity.
        :returns: The current stub tensor instance.
        """
        return self

    def __bool__(self) -> bool:
        """Return truthiness based on whether any backing data exists.

        :returns: ``True`` when any backing data is present.
        """
        return bool(self._data)


def build_torch_stubs() -> Tuple[Any, Any, Any]:
    """Construct minimal stand-ins for ``torch``, ``torch.nn`` and ``torch.optim``.

    :returns: Tuple of stub modules ``(torch_stub, nn_stub, optim_stub)``.
    """

    class _CudaStub:
        """Subset of :mod:`torch.cuda` used when PyTorch is unavailable."""

        CudaError = RuntimeError

        @staticmethod
        def is_available() -> bool:
            """Return ``False`` to mimic the absence of CUDA support.

            :returns: ``False`` because the stub never exposes CUDA support.
            """
            return False

        @staticmethod
        def device_count() -> int:
            """Return ``0`` CUDA devices to match CPU-only environments.

            :returns: ``0`` to indicate no CUDA devices are available.
            """
            return 0

    class _ModuleStub:
        """Minimal :class:`torch.nn.Module` replacement for documentation."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Accept arbitrary arguments for API compatibility.

            :param _args: Positional arguments forwarded from callers.
            :param _kwargs: Keyword arguments forwarded from callers.
            :returns: ``None``. Stores no internal state.
            """
            del _args, _kwargs

        def register_buffer(self, *_args: Any, **_kwargs: Any) -> None:
            """Ignore buffer registration in the stub implementation.

            :param _args: Positional arguments forwarded from callers.
            :param _kwargs: Keyword arguments forwarded from callers.
            :returns: ``None`` because no buffers are tracked.
            """
            del _args, _kwargs

        def parameters(self) -> Sequence[Any]:
            """Return an empty iterable of learnable parameters.

            :returns: Empty list for compatibility with optimisation loops.
            """
            return []

    class _ParameterStub:
        """Emulate :class:`torch.nn.Parameter` storage."""

        def __init__(self, value: Any) -> None:
            """Persist the provided value without tensor semantics.

            :param value: Object that should be wrapped by the stub.
            :returns: ``None``. Stores the value for later inspection.
            """
            self.value = value

        def to(self, *_args: Any, **_kwargs: Any) -> "_ParameterStub":
            """Return ``self`` to align with the real ``Parameter.to`` API.

            :param _args: Positional arguments forwarded from callers.
            :param _kwargs: Keyword arguments forwarded from callers.
            :returns: The current parameter stub.
            """
            del _args, _kwargs
            return self

        def detach(self) -> "_ParameterStub":
            """Return ``self`` to mirror ``torch.nn.Parameter.detach``.

            :returns: The current parameter stub.
            """
            return self

    class _AdamStub:
        """Tiny drop-in replacement for :class:`torch.optim.Adam`."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            """Discard configuration arguments while keeping the signature.

            :param _args: Positional arguments forwarded from callers.
            :param _kwargs: Keyword arguments forwarded from callers.
            :returns: ``None``. Keeps no optimiser state.
            """
            del _args, _kwargs

        def zero_grad(self, *_args: Any, **_kwargs: Any) -> None:
            """No-op gradient reset for the stub optimiser.

            :param _args: Positional arguments forwarded from callers.
            :param _kwargs: Keyword arguments forwarded from callers.
            :returns: ``None`` because no gradients are tracked.
            """
            del _args, _kwargs

        def step(self) -> None:
            """No-op optimiser step.

            :returns: ``None`` because parameters are not updated.
            """

        def state_dict(self) -> Dict[str, Any]:
            """Return an empty optimiser state.

            :returns: State dictionary compatible with checkpoint callers.
            """
            return {}

    def _make_tensor(*args: Any, **kwargs: Any) -> TensorStub:
        """Build a :class:`TensorStub` from standard ``torch.tensor`` inputs.

        :param args: Positional arguments matching the real API.
        :param kwargs: Keyword arguments matching the real API.
        :returns: New :class:`TensorStub` instance.
        """
        data = None
        if args:
            candidate = args[0]
            if isinstance(candidate, int):
                data = [0.0] * candidate
            elif isinstance(candidate, (list, tuple)):
                data = list(candidate)
        return TensorStub(data, device=kwargs.get("device"))

    torch_stub = types.SimpleNamespace(
        Tensor=TensorStub,
        tensor=_make_tensor,
        zeros=lambda *args, **kwargs: TensorStub(
            [0.0] * int(args[0]) if args else [], device=kwargs.get("device")
        ),
        zeros_like=lambda *_args, **_kwargs: TensorStub(),
        ones=lambda *args, **kwargs: TensorStub(
            [1.0] * int(args[0]) if args else [], device=kwargs.get("device")
        ),
        ones_like=lambda *_args, **_kwargs: TensorStub(),
        stack=lambda *_args, **_kwargs: TensorStub(),
        softmax=lambda *_args, **_kwargs: TensorStub(),
        no_grad=lambda: (lambda func: func),
        isfinite=lambda *_args, **_kwargs: types.SimpleNamespace(all=lambda: True),
        allclose=lambda *_args, **_kwargs: False,
        cuda=_CudaStub(),
        device=lambda spec: spec,
        log=lambda *_args, **_kwargs: TensorStub(),
        float32="float32",
    )
    nn_stub = types.SimpleNamespace(
        Module=_ModuleStub,
        Parameter=_ParameterStub,
    )
    optim_stub = types.SimpleNamespace(Adam=_AdamStub)

    return torch_stub, nn_stub, optim_stub

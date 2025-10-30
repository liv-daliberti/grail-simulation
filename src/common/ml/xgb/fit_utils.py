#!/usr/bin/env python
# Copyright 2025 The Grail Simulation Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helpers to harmonise XGBoost estimator.fit kwargs across versions.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict


def harmonize_fit_kwargs(
    estimator: Any,
    fit_kwargs: Dict[str, Any],
    *,
    has_eval: bool,
    early_stopping_rounds: int = 50,
) -> None:
    """
    Prune unsupported kwargs and attach legacy early stopping when applicable.

    - Removes "callbacks" when the installed estimator does not support it.
    - Adds "early_stopping_rounds" as a fallback when callbacks aren't available.
    - Drops None-valued and unsupported keys based on the signature.

    :param estimator: XGBoost estimator exposing a ``fit`` method.
    :param fit_kwargs: Keyword arguments forwarded to ``estimator.fit``.
    :param has_eval: Indicates whether evaluation data is provided.
    :param early_stopping_rounds: Early stopping patience used when callbacks are unavailable.
    :returns: ``None``.
    """

    try:
        fit_sig = inspect.signature(getattr(estimator, "fit"))
        supports_callbacks = "callbacks" in fit_sig.parameters

        callbacks = fit_kwargs.get("callbacks")
        if not supports_callbacks:
            fit_kwargs.pop("callbacks", None)

        has_es_callback = False
        if supports_callbacks and isinstance(callbacks, (list, tuple)):
            for callback in callbacks:
                name = type(callback).__name__
                if "EarlyStopping" in str(name):
                    has_es_callback = True
                    break

        if has_eval and (not has_es_callback) and (
            "early_stopping_rounds" in fit_sig.parameters
        ):
            fit_kwargs["early_stopping_rounds"] = early_stopping_rounds

        # Finally, drop keys not supported by the installed signature or None values.
        supported_params = set(fit_sig.parameters)
        for key in list(fit_kwargs.keys()):
            if fit_kwargs.get(key) is None or key not in supported_params:
                fit_kwargs.pop(key, None)
    except (ValueError, TypeError, AttributeError):  # pragma: no cover - defensive
        # Best effort; if inspection fails, at least avoid passing None values.
        for key in [k for k, v in list(fit_kwargs.items()) if v is None]:
            fit_kwargs.pop(key, None)


__all__ = ["harmonize_fit_kwargs"]

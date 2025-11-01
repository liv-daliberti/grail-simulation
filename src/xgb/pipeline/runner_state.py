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

"""Mutable runner state container extracted from the pipeline runner.

This module provides the ``RunnerState`` dataclass to centralise mutable
state used by the pipeline orchestration. It reduces the size of the
``runner.py`` module and makes the state easier to import and test
independently. Property implementations live in small mixin modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ._state_mixins_flags_config import _FlagsMixin, _ConfigMixin
from ._state_mixins_data_contexts import _DataMixin, _ContextsMixin
from ._state_mixins_planning_results import _PlanningMixin, _ResultsMixin


@dataclass
class RunnerState(
    _FlagsMixin,
    _ConfigMixin,
    _DataMixin,
    _ContextsMixin,
    _PlanningMixin,
    _ResultsMixin,
):
    """Mutable container for pipeline state to reduce locals in runner.

    Only a small number of actual instance attributes are declared to keep
    pylint's attribute counts low. All public fields are exposed via
    properties implemented in mixins and backed by a single internal store.
    """

    args: Any
    extra_cli: Sequence[str]

    # Single internal storage mapping grouping all categories of state.
    _state: dict | None = None

    def __post_init__(self) -> None:  # type: ignore[override]
        """Initialise the internal storage mapping on first construction."""
        if self._state is None:
            self._state = {
                "flags": {},
                "cfg": {},
                "data": {},
                "ctx": {},
                "plan": {},
                "results": {},
            }


__all__ = ["RunnerState"]

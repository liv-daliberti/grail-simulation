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

"""RunnerState mixins for data and context properties."""

from __future__ import annotations


class _DataMixin:
    # Data
    @property
    def study_specs(self):  # type: ignore[override]
        """Resolved studies and issues specification for the run."""
        data = self._state.setdefault("data", {})  # type: ignore[attr-defined]
        return data.get("study_specs")

    @study_specs.setter
    def study_specs(self, value) -> None:  # type: ignore[override]
        """Set the resolved study specifications."""
        self._state.setdefault("data", {})["study_specs"] = value  # type: ignore[attr-defined]

    @property
    def study_tokens_tuple(self) -> tuple[str, ...]:  # type: ignore[override]
        """Selected study tokens as an immutable tuple."""
        data = self._state.setdefault("data", {})  # type: ignore[attr-defined]
        return data.get("study_tokens_tuple", ())

    @study_tokens_tuple.setter
    def study_tokens_tuple(self, value: tuple[str, ...]) -> None:  # type: ignore[override]
        """Set the selected study tokens."""
        data = self._state.setdefault("data", {})  # type: ignore[attr-defined]
        data["study_tokens_tuple"] = value

    @property
    def extra_fields_tuple(self) -> tuple[str, ...]:  # type: ignore[override]
        """Additional text fields to include as features."""
        data = self._state.setdefault("data", {})  # type: ignore[attr-defined]
        return data.get("extra_fields_tuple", ())

    @extra_fields_tuple.setter
    def extra_fields_tuple(self, value: tuple[str, ...]) -> None:  # type: ignore[override]
        """Set additional text fields to include as features."""
        data = self._state.setdefault("data", {})  # type: ignore[attr-defined]
        data["extra_fields_tuple"] = value

    @property
    def configs(self):  # type: ignore[override]
        """Sweep configuration grid for model training and evaluation."""
        data = self._state.setdefault("data", {})  # type: ignore[attr-defined]
        return data.get("configs")

    @configs.setter
    def configs(self, value) -> None:  # type: ignore[override]
        """Set the hyperparameter configuration grid."""
        self._state.setdefault("data", {})["configs"] = value  # type: ignore[attr-defined]


class _ContextsMixin:
    # Contexts
    @property
    def sweep_context(self):  # type: ignore[override]
        """Execution context for Next Video sweeps."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        return ctx.get("sweep_context")

    @sweep_context.setter
    def sweep_context(self, value) -> None:  # type: ignore[override]
        """Set the Next Video sweep execution context."""
        self._state.setdefault("ctx", {})["sweep_context"] = value  # type: ignore[attr-defined]

    @property
    def opinion_sweep_context(self):  # type: ignore[override]
        """Execution context for Opinion sweeps."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        return ctx.get("opinion_sweep_context")

    @opinion_sweep_context.setter
    def opinion_sweep_context(self, value) -> None:  # type: ignore[override]
        """Set the Opinion sweep execution context."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        ctx["opinion_sweep_context"] = value

    @property
    def opinion_stage_config(self):  # type: ignore[override]
        """Stage configuration for Opinion reporting/finalization."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        return ctx.get("opinion_stage_config")

    @opinion_stage_config.setter
    def opinion_stage_config(self, value) -> None:  # type: ignore[override]
        """Set the Opinion stage configuration."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        ctx["opinion_stage_config"] = value

    @property
    def final_eval_context(self):  # type: ignore[override]
        """Context for the final evaluation stage of Next Video."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        return ctx.get("final_eval_context")

    @final_eval_context.setter
    def final_eval_context(self, value) -> None:  # type: ignore[override]
        """Set the final evaluation context for Next Video."""
        ctx = self._state.setdefault("ctx", {})  # type: ignore[attr-defined]
        ctx["final_eval_context"] = value


__all__ = ["_DataMixin", "_ContextsMixin"]

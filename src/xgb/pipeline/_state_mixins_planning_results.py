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

"""RunnerState mixins for planning and results properties."""

from __future__ import annotations


class _PlanningMixin:
    # Planning
    @property
    def planned_slate_tasks(self):  # type: ignore[override]
        """Prepared Next Video tasks ready for execution."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        return plan.get("planned_slate_tasks")

    @planned_slate_tasks.setter
    def planned_slate_tasks(self, value) -> None:  # type: ignore[override]
        """Set prepared Next Video tasks."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        plan["planned_slate_tasks"] = value

    @property
    def cached_slate_planned(self):  # type: ignore[override]
        """Cached Next Video outcomes found during planning."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        return plan.get("cached_slate_planned")

    @cached_slate_planned.setter
    def cached_slate_planned(self, value) -> None:  # type: ignore[override]
        """Set cached Next Video outcomes found during planning."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        plan["cached_slate_planned"] = value

    @property
    def planned_opinion_tasks(self):  # type: ignore[override]
        """Prepared Opinion tasks ready for execution."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        return plan.get("planned_opinion_tasks")

    @planned_opinion_tasks.setter
    def planned_opinion_tasks(self, value) -> None:  # type: ignore[override]
        """Set prepared Opinion tasks."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        plan["planned_opinion_tasks"] = value

    @property
    def cached_opinion_planned(self):  # type: ignore[override]
        """Cached Opinion outcomes found during planning."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        return plan.get("cached_opinion_planned")

    @cached_opinion_planned.setter
    def cached_opinion_planned(self, value) -> None:  # type: ignore[override]
        """Set cached Opinion outcomes found during planning."""
        plan = self._state.setdefault("plan", {})  # type: ignore[attr-defined]
        plan["cached_opinion_planned"] = value


class _ResultsMixin:
    # Execution inputs/outputs
    @property
    def pending_slate_tasks(self):  # type: ignore[override]
        """Pending Next Video tasks to execute for this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("pending_slate_tasks")

    @pending_slate_tasks.setter
    def pending_slate_tasks(self, value) -> None:  # type: ignore[override]
        """Set pending Next Video tasks to execute for this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["pending_slate_tasks"] = value

    @property
    def cached_slate_outcomes(self):  # type: ignore[override]
        """Cached outcomes for Next Video sweeps used in this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("cached_slate_outcomes")

    @cached_slate_outcomes.setter
    def cached_slate_outcomes(self, value) -> None:  # type: ignore[override]
        """Set cached outcomes for Next Video sweeps used in this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["cached_slate_outcomes"] = value

    @property
    def pending_opinion_tasks(self):  # type: ignore[override]
        """Pending Opinion tasks to execute for this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("pending_opinion_tasks")

    @pending_opinion_tasks.setter
    def pending_opinion_tasks(self, value) -> None:  # type: ignore[override]
        """Set pending Opinion tasks to execute for this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["pending_opinion_tasks"] = value

    @property
    def cached_opinion_outcomes(self):  # type: ignore[override]
        """Cached outcomes for Opinion sweeps used in this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("cached_opinion_outcomes")

    @cached_opinion_outcomes.setter
    def cached_opinion_outcomes(self, value) -> None:  # type: ignore[override]
        """Set cached outcomes for Opinion sweeps used in this stage."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["cached_opinion_outcomes"] = value

    @property
    def executed_slate_outcomes(self):  # type: ignore[override]
        """Outcomes produced by executing Next Video tasks in this run."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("executed_slate_outcomes")

    @executed_slate_outcomes.setter
    def executed_slate_outcomes(self, value) -> None:  # type: ignore[override]
        """Set outcomes produced by executing Next Video tasks in this run."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["executed_slate_outcomes"] = value

    @property
    def executed_opinion_outcomes(self):  # type: ignore[override]
        """Outcomes produced by executing Opinion tasks in this run."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("executed_opinion_outcomes")

    @executed_opinion_outcomes.setter
    def executed_opinion_outcomes(self, value) -> None:  # type: ignore[override]
        """Set outcomes produced by executing Opinion tasks in this run."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["executed_opinion_outcomes"] = value

    @property
    def outcomes(self):  # type: ignore[override]
        """Merged outcomes combining cached and executed results."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("outcomes")

    @outcomes.setter
    def outcomes(self, value) -> None:  # type: ignore[override]
        """Set the merged outcomes for the selected tasks."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["outcomes"] = value

    @property
    def opinion_sweep_outcomes(self):  # type: ignore[override]
        """Outcomes for Opinion sweeps only."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("opinion_sweep_outcomes")

    @opinion_sweep_outcomes.setter
    def opinion_sweep_outcomes(self, value) -> None:  # type: ignore[override]
        """Set outcomes for Opinion sweeps only."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["opinion_sweep_outcomes"] = value

    @property
    def selections(self):  # type: ignore[override]
        """Selected configurations after evaluation for Next Video."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("selections")

    @selections.setter
    def selections(self, value) -> None:  # type: ignore[override]
        """Set selected configurations after evaluation for Next Video."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["selections"] = value

    @property
    def opinion_selections(self):  # type: ignore[override]
        """Selected configurations after evaluation for Opinion."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        return results.get("opinion_selections")

    @opinion_selections.setter
    def opinion_selections(self, value) -> None:  # type: ignore[override]
        """Set selected configurations after evaluation for Opinion."""
        results = self._state.setdefault("results", {})  # type: ignore[attr-defined]
        results["opinion_selections"] = value


__all__ = ["_PlanningMixin", "_ResultsMixin"]

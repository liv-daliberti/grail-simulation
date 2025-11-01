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

"""Pipeline runner entry points and top-level orchestration."""

from __future__ import annotations

import logging
import os
import importlib as _importlib
from pathlib import Path
from typing import Any, List, Sequence, Optional
from dataclasses import dataclass, field

from common.prompts.docs import merge_default_extra_fields

from .cli import (
    _parse_args,
    _split_tokens,
)
from .context import FinalEvalContext
from .settings import (
    build_base_cli,
    build_paths,
    build_vectorizer_configs,
    determine_jobs,
    determine_tree_method,
    log_reuse_flags,
    resolve_reuse_flags,
)
from .stages import (
    build_contexts,
    finalize_and_report,
    handle_plan_or_dry_run,
    handle_sweeps_stage,
    maybe_execute_full_stage,
    maybe_reports_stage,
    merge_and_validate_outcomes,
    prepare_pending_and_cached,
    select_and_log_configs,
    validate_missing_metrics,
)
from .word2vec_cache import apply_word2vec_workers_override


LOGGER = logging.getLogger("xgb.pipeline")


@dataclass
class ReuseFlags:
    """Reuse controls and metadata for reusing artifacts across runs."""
    sweeps: bool = False
    final: bool = False
    sweeps_source: Optional[str] = None
    final_source: Optional[str] = None


@dataclass
class Flags:
    """Top-level pipeline flags: task toggles, allowance, and stage."""
    run_next_video: bool = True
    run_opinion: bool = True
    allow_incomplete: bool = True
    stage: str = "full"
    reuse: ReuseFlags = field(default_factory=ReuseFlags)


@dataclass
class Env:
    """Environment and parallelism settings resolved from CLI."""
    paths: Any | None = None
    jobs: int = 1


@dataclass
class Inputs:
    """Dataset and text-field selection resolved from CLI and defaults."""
    study_specs: Any | None = None
    study_tokens_tuple: tuple[str, ...] = field(default_factory=tuple)
    extra_fields_tuple: tuple[str, ...] = field(default_factory=tuple)
    base_cli: Sequence[str] = field(default_factory=tuple)


@dataclass
class ConfigBundle:
    """Sweep grid and vectorizer configurations bundled together."""
    grid: Any | None = None
    vectorizers: Any | None = None


@dataclass
class RunnerConfig:
    """Aggregate configuration grouping flags, environment, inputs and configs."""
    flags: Flags = field(default_factory=Flags)
    env: Env = field(default_factory=Env)
    inputs: Inputs = field(default_factory=Inputs)
    configs: ConfigBundle = field(default_factory=ConfigBundle)


@dataclass
class RunnerContexts:
    """Execution contexts used across stages."""

    sweep_context: Any | None = None
    opinion_sweep_context: Any | None = None
    opinion_stage_config: Any | None = None
    final_eval_context: Any | None = None


@dataclass
class RunnerPlan:
    """Planned tasks discovered in the planning stage."""

    planned_slate_tasks: list = field(default_factory=list)
    cached_slate_planned: list = field(default_factory=list)
    planned_opinion_tasks: list = field(default_factory=list)
    cached_opinion_planned: list = field(default_factory=list)


@dataclass
class Pending:
    """Pending tasks to execute for each flow."""
    slate_tasks: list = field(default_factory=list)
    opinion_tasks: list = field(default_factory=list)


@dataclass
class Cached:
    """Cached outcomes already available for each flow."""
    slate_outcomes: list = field(default_factory=list)
    opinion_outcomes: list = field(default_factory=list)


@dataclass
class Executed:
    """Outcomes produced by task execution in this run."""
    slate_outcomes: list = field(default_factory=list)
    opinion_outcomes: list = field(default_factory=list)


@dataclass
class Merged:
    """Merged outcomes combining cached and executed results."""
    outcomes: list = field(default_factory=list)
    opinion_sweep_outcomes: list = field(default_factory=list)


@dataclass
class Choices:
    """Final configuration selections for each flow."""
    selections: dict = field(default_factory=dict)
    opinion_selections: dict = field(default_factory=dict)


@dataclass
class RunnerResults:
    """Execution results and selections accumulated across stages."""

    pending: Pending = field(default_factory=Pending)
    cached: Cached = field(default_factory=Cached)
    executed: Executed = field(default_factory=Executed)
    merged: Merged = field(default_factory=Merged)
    choices: Choices = field(default_factory=Choices)


class RunnerState:
    """Mutable state container used by the pipeline runner.

    Groups attributes into a few cohesive dataclasses to keep the surface
    tight and focused while remaining easy to test and monkeypatch.
    """

    def __init__(self, *, args: Any, extra_cli: Sequence[str]) -> None:
        self.args: Any = args
        self.extra_cli: List[str] = list(extra_cli)
        self.cfg = RunnerConfig()
        self.contexts = RunnerContexts()
        self.plan = RunnerPlan()
        self.results = RunnerResults()

    def reset_plan(self) -> None:
        """Clear planned tasks and cached-planned outcomes."""
        self.plan = RunnerPlan()

    def reset_results(self) -> None:
        """Clear transient results and selections."""
        self.results = RunnerResults()

    def selected_tasks_summary(self) -> List[str]:
        """Return a human-readable list of selected task names."""
        pairs = (
            ("next_video", self.cfg.flags.run_next_video),
            ("opinion", self.cfg.flags.run_opinion),
        )
        return [name for name, enabled in pairs if enabled]



def _configure_basics(state: RunnerState) -> None:
    """Compute paths, jobs and set cache envs."""
    state.cfg.env.paths = build_paths(state.args)
    state.cfg.env.jobs = determine_jobs(state.args)

    # Ensure deterministic HuggingFace caches
    # Be robust if path resolution is unexpectedly None in certain test setups.
    if state.cfg.env.paths is not None and getattr(state.cfg.env.paths, "cache_dir", None):
        os.environ.setdefault("HF_DATASETS_CACHE", state.cfg.env.paths.cache_dir)
        os.environ.setdefault("HF_HOME", state.cfg.env.paths.cache_dir)


def _determine_allow_incomplete(state: RunnerState) -> None:
    """Resolve allow_incomplete flag from CLI or environment."""
    allow_incomplete = getattr(state.args, "allow_incomplete", True)
    env_allow = os.environ.get("XGB_ALLOW_INCOMPLETE")
    if env_allow is not None:
        allow_incomplete = env_allow.lower() not in {"0", "false", "no"}
    state.cfg.flags.allow_incomplete = allow_incomplete


def _resolve_dataset_specs(state: RunnerState) -> None:
    """Resolve requested issues/studies and additional text fields."""
    issue_tokens = _split_tokens(state.args.issues)
    study_tokens = _split_tokens(state.args.studies)
    state.cfg.inputs.study_tokens_tuple = tuple(study_tokens)
    # Resolve via top-level xgb.pipeline to allow monkeypatching in tests.
    resolver = getattr(_importlib.import_module("xgb.pipeline"), "_resolve_study_specs")
    state.cfg.inputs.study_specs = resolver(
        dataset=state.cfg.env.paths.dataset,
        cache_dir=state.cfg.env.paths.cache_dir,
        requested_issues=issue_tokens,
        requested_studies=study_tokens,
        allow_incomplete=state.cfg.flags.allow_incomplete,
    )
    extra_fields = merge_default_extra_fields(_split_tokens(state.args.extra_text_fields))
    state.cfg.inputs.extra_fields_tuple = tuple(extra_fields)


def _configure_xgb_and_base_cli(state: RunnerState) -> None:
    """Log parallelism and configure XGBoost + base CLI."""
    LOGGER.info("Parallel sweep jobs: %d", state.cfg.env.jobs)
    state.args.tree_method = determine_tree_method(state.args)
    LOGGER.info("Using XGBoost tree_method=%s.", state.args.tree_method)
    state.cfg.inputs.base_cli = build_base_cli(
        state.args, state.cfg.env.paths, list(state.cfg.inputs.extra_fields_tuple)
    )


def _configure_stage_and_reuse(state: RunnerState) -> None:
    """Resolve stage and reuse settings, applying side effects as needed.

    Parameters
    ----------
    state : RunnerState
        Mutable runner state holding CLI args and computed values.

    Returns
    -------
    None
        Updates ``state.stage`` and all reuse-related fields in place and
        applies word2vec worker overrides when reusing or finalizing.
    """
    state.cfg.flags.stage = getattr(state.args, "stage", "full")
    (
        state.cfg.flags.reuse.sweeps,
        state.cfg.flags.reuse.final,
        state.cfg.flags.reuse.sweeps_source,
        state.cfg.flags.reuse.final_source,
    ) = resolve_reuse_flags(state.args)

    if state.cfg.flags.stage in {"finalize", "reports"} or state.cfg.flags.reuse.sweeps:
        apply_word2vec_workers_override(state.args, paths=state.cfg.env.paths)

    log_reuse_flags(
        state.cfg.flags.reuse.sweeps,
        state.cfg.flags.reuse.final,
        state.cfg.flags.reuse.sweeps_source,
        state.cfg.flags.reuse.final_source,
    )


def _build_configs(state: RunnerState) -> None:
    """Construct the sweep configuration grid from CLI arguments.

    Parameters
    ----------
    state : RunnerState
        Mutable runner state to receive the built configuration grid.

    Returns
    -------
    None
    """
    # Resolve via top-level xgb.pipeline to allow monkeypatching in tests.
    builder = getattr(_importlib.import_module("xgb.pipeline"), "_build_sweep_configs")
    state.cfg.configs.grid = builder(state.args)


def _build_vectorizers(state: RunnerState) -> None:
    """Build vectorizer configuration bundle from CLI arguments.

    Parameters
    ----------
    state : RunnerState
        Mutable runner state to receive text vectorizer configurations.

    Returns
    -------
    None
    """
    state.cfg.configs.vectorizers = build_vectorizer_configs(state.args)


def _build_contexts_for_tasks(state: RunnerState) -> None:
    """Create execution contexts required to prepare tasks for each flow.

    Parameters
    ----------
    state : RunnerState
        Mutable runner state providing inputs and receiving built contexts.

    Returns
    -------
    None
    """
    (
        state.contexts.sweep_context,
        state.contexts.opinion_sweep_context,
        state.contexts.opinion_stage_config,
    ) = build_contexts(
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        args=state.args,
        paths=state.cfg.env.paths,
        jobs=state.cfg.env.jobs,
        base_cli=state.cfg.inputs.base_cli,
        extra_cli=state.extra_cli,
        max_features_value=(
            state.cfg.configs.vectorizers.max_features_value
            if state.cfg.configs.vectorizers
            else None
        ),
        tfidf_config=(
            state.cfg.configs.vectorizers.tfidf_config if state.cfg.configs.vectorizers else None
        ),
        word2vec_config=(
            state.cfg.configs.vectorizers.word2vec_config if state.cfg.configs.vectorizers else None
        ),
        sentence_transformer_config=(
            state.cfg.configs.vectorizers.sentence_transformer_config
            if state.cfg.configs.vectorizers
            else None
        ),
        word2vec_model_base=(
            state.cfg.configs.vectorizers.word2vec_model_base
            if state.cfg.configs.vectorizers
            else None
        ),
        reuse_final=state.cfg.flags.reuse.final,
        extra_fields_tuple=state.cfg.inputs.extra_fields_tuple,
        study_tokens_tuple=state.cfg.inputs.study_tokens_tuple,
    )


def _plan_tasks(state: RunnerState) -> None:
    """Build task plans for both Next Video and Opinion flows.

    Parameters
    ----------
    state : RunnerState
        Mutable runner state providing inputs and receiving task plans.

    Returns
    -------
    None
    """
    state.plan.planned_slate_tasks = []
    state.plan.cached_slate_planned = []
    if state.cfg.flags.run_next_video and state.contexts.sweep_context is not None:
        pipeline_mod = _importlib.import_module("xgb.pipeline")
        prepare_sweep = getattr(
            pipeline_mod, "_prepare_sweep_tasks"
        )  # type: ignore[attr-defined]
        (
            state.plan.planned_slate_tasks,
            state.plan.cached_slate_planned,
        ) = prepare_sweep(
            studies=state.cfg.inputs.study_specs,
            configs=state.cfg.configs.grid,
            context=state.contexts.sweep_context,
            reuse_existing=state.cfg.flags.reuse.sweeps,
        )

    state.plan.planned_opinion_tasks = []
    state.plan.cached_opinion_planned = []
    if state.cfg.flags.run_opinion and state.contexts.opinion_sweep_context is not None:
        pipeline_mod = _importlib.import_module("xgb.pipeline")
        prepare_opinion = getattr(
            pipeline_mod, "_prepare_opinion_sweep_tasks"
        )  # type: ignore[attr-defined]
        (
            state.plan.planned_opinion_tasks,
            state.plan.cached_opinion_planned,
        ) = prepare_opinion(
            studies=state.cfg.inputs.study_specs,
            configs=state.cfg.configs.grid,
            context=state.contexts.opinion_sweep_context,
            reuse_existing=state.cfg.flags.reuse.sweeps,
        )


def _maybe_handle_plan_or_dry_run(state: RunnerState) -> bool:
    """Handle ``plan`` and ``--dry-run`` stages with early exit.

    Parameters
    ----------
    state : RunnerState
        Current runner state including planned tasks.

    Returns
    -------
    bool
        ``True`` if the function handled the request and the pipeline should
        exit early; otherwise ``False``.
    """
    return handle_plan_or_dry_run(
        stage=state.cfg.flags.stage,
        args=state.args,
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        planned_slate_tasks=state.plan.planned_slate_tasks or [],
        cached_slate_planned=state.plan.cached_slate_planned or [],
        planned_opinion_tasks=state.plan.planned_opinion_tasks or [],
        cached_opinion_planned=state.plan.cached_opinion_planned or [],
    )


def _maybe_handle_sweeps_stage(state: RunnerState) -> bool:
    """Prepare and optionally execute the sweep stage, with early exit.

    Parameters
    ----------
    state : RunnerState
        Current runner state including planned tasks and configs.

    Returns
    -------
    bool
        ``True`` if the sweeps stage was targeted and handled; otherwise
        ``False`` to continue to subsequent stages.
    """
    # Import lazily via importlib to avoid static cycles and allow monkeypatching
    _prepare = getattr(
        _importlib.import_module("xgb.pipeline"),
        "prepare_sweep_execution",
    )  # type: ignore[attr-defined]

    return handle_sweeps_stage(
        stage=state.cfg.flags.stage,
        args=state.args,
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        planned_slate_tasks=state.plan.planned_slate_tasks or [],
        cached_slate_planned=state.plan.cached_slate_planned or [],
        planned_opinion_tasks=state.plan.planned_opinion_tasks or [],
        cached_opinion_planned=state.plan.cached_opinion_planned or [],
        prepare=_prepare,
    )


def _prepare_pending_and_execute(state: RunnerState) -> None:
    """Prepare pending tasks, execute as needed, and compute selections.

    Parameters
    ----------
    state : RunnerState
        Current runner state used both as input and output container.

    Returns
    -------
    None
        Mutates ``state`` to include pending tasks, outcomes, and selections.
    """
    (
        state.results.pending.slate_tasks,
        state.results.cached.slate_outcomes,
        state.results.pending.opinion_tasks,
        state.results.cached.opinion_outcomes,
    ) = prepare_pending_and_cached(
        stage=state.cfg.flags.stage,
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        sweep_context=state.contexts.sweep_context,
        opinion_sweep_context=state.contexts.opinion_sweep_context,
        study_specs=state.cfg.inputs.study_specs,
        configs=state.cfg.configs.grid,
        reuse_sweeps=state.cfg.flags.reuse.sweeps,
    )

    validate_missing_metrics(
        stage=state.cfg.flags.stage,
        allow_incomplete=state.cfg.flags.allow_incomplete,
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        pending_slate_tasks=state.results.pending.slate_tasks,
        pending_opinion_tasks=state.results.pending.opinion_tasks,
    )

    (
        state.results.executed.slate_outcomes,
        state.results.executed.opinion_outcomes,
    ) = maybe_execute_full_stage(
        stage=state.cfg.flags.stage,
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        pending_slate_tasks=state.results.pending.slate_tasks,
        pending_opinion_tasks=state.results.pending.opinion_tasks,
        jobs=state.cfg.env.jobs,
    )

    (
        state.results.merged.outcomes,
        state.results.merged.opinion_sweep_outcomes,
    ) = merge_and_validate_outcomes(
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        allow_incomplete=state.cfg.flags.allow_incomplete,
        cached_slate_outcomes=state.results.cached.slate_outcomes,
        executed_slate_outcomes=state.results.executed.slate_outcomes,
        cached_opinion_outcomes=state.results.cached.opinion_outcomes,
        executed_opinion_outcomes=state.results.executed.opinion_outcomes,
    )

    (
        state.results.choices.selections,
        state.results.choices.opinion_selections,
    ) = select_and_log_configs(
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        allow_incomplete=state.cfg.flags.allow_incomplete,
        outcomes=state.results.merged.outcomes,
        opinion_sweep_outcomes=state.results.merged.opinion_sweep_outcomes,
    )


def _maybe_reports_or_finalize(state: RunnerState) -> bool:
    """Run reports or finalization stages depending on configuration.

    Parameters
    ----------
    state : RunnerState
        Current runner state providing selections and outcomes.

    Returns
    -------
    bool
        ``True`` if the pipeline should exit after report generation;
        otherwise ``False`` after finalization.
    """
    if state.cfg.flags.run_next_video:
        state.contexts.final_eval_context = FinalEvalContext(
            base_cli=state.cfg.inputs.base_cli,
            extra_cli=state.extra_cli,
            out_dir=state.cfg.env.paths.next_video_dir,
            tree_method=state.args.tree_method,
            save_model_dir=Path(state.args.save_model_dir) if state.args.save_model_dir else None,
            reuse_existing=state.cfg.flags.reuse.final,
        )

    if maybe_reports_stage(
        stage=state.cfg.flags.stage,
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        allow_incomplete=state.cfg.flags.allow_incomplete,
        final_eval_context=state.contexts.final_eval_context,
        opinion_stage_config=state.contexts.opinion_stage_config,
        selections=state.results.choices.selections,
        opinion_selections=state.results.choices.opinion_selections,
        study_specs=state.cfg.inputs.study_specs,
        paths=state.cfg.env.paths,
        outcomes=state.results.merged.outcomes,
        opinion_sweep_outcomes=state.results.merged.opinion_sweep_outcomes,
    ):
        return True

    finalize_and_report(
        run_next_video=state.cfg.flags.run_next_video,
        run_opinion=state.cfg.flags.run_opinion,
        allow_incomplete=state.cfg.flags.allow_incomplete,
        final_eval_context=state.contexts.final_eval_context,
        opinion_stage_config=state.contexts.opinion_stage_config,
        selections=state.results.choices.selections,
        opinion_selections=state.results.choices.opinion_selections,
        outcomes=state.results.merged.outcomes,
        opinion_sweep_outcomes=state.results.merged.opinion_sweep_outcomes,
        paths=state.cfg.env.paths,
        study_specs=state.cfg.inputs.study_specs,
    )
    return False


def _run_pipeline(args, extra_cli: Sequence[str]) -> None:
    """Run the end-to-end pipeline for the selected tasks.

    Parameters
    ----------
    args : argparse.Namespace | Any
        Parsed CLI arguments controlling the run.
    extra_cli : Sequence[str]
        Additional CLI arguments to pass through to subcommands.

    Returns
    -------
    None
    """
    # Task selection and early exit
    run_next_video = getattr(args, "run_next_video", True)
    run_opinion = getattr(args, "run_opinion", True)
    if not run_next_video and not run_opinion:
        LOGGER.warning("No tasks selected; exiting.")
        return

    state = RunnerState(
        args=args,
        extra_cli=extra_cli,
    )
    # Set initial task selection flags in config
    state.cfg.flags.run_next_video = run_next_video
    state.cfg.flags.run_opinion = run_opinion

    # Basics and inputs
    _configure_basics(state)
    _determine_allow_incomplete(state)
    _resolve_dataset_specs(state)
    # Logging task selection
    task_log: List[str] = state.selected_tasks_summary()
    LOGGER.info("Selected pipeline tasks: %s.", ", ".join(task_log))

    # Config and stage setup
    _configure_xgb_and_base_cli(state)
    _configure_stage_and_reuse(state)
    _build_configs(state)
    _build_vectorizers(state)
    _build_contexts_for_tasks(state)

    # Plan and maybe stop early
    _plan_tasks(state)
    if _maybe_handle_plan_or_dry_run(state):
        return
    if _maybe_handle_sweeps_stage(state):
        return

    # Execute, report, finalize
    _prepare_pending_and_execute(state)
    if _maybe_reports_or_finalize(state):
        return


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point orchestrating the full XGBoost workflow."""

    args, extra_cli = _parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    _run_pipeline(args, extra_cli)


__all__ = ["main"]

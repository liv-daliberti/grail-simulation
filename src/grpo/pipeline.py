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

"""CLI entry point for evaluating and reporting on GRPO baselines."""

from __future__ import annotations

import argparse
import faulthandler
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from common.opinion import DEFAULT_SPECS

try:
    # Evaluation pulls in Pydantic models that attach ``repr``/``frozen`` metadata
    # via ``Field(...)``. Pydantic 2.x warns about those attributes even though
    # the originating libraries expect them to be ignored. Filter to keep logs clean.
    from pydantic.warnings import UnsupportedFieldAttributeWarning
except ImportError:  # pragma: no cover - Pydantic may not be available in minimal envs
    UnsupportedFieldAttributeWarning = None  # type: ignore[assignment]
else:
    warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
from common.rlhf.reports import ReportOptions

from . import DEFAULT_DATASET_PATH, DEFAULT_EVAL_SPLIT
from .config import DEFAULT_SYSTEM_PROMPT, OPINION_SYSTEM_PROMPT, repo_root as _repo_root
from .next_video import (
    FilterSelection,
    NextVideoEvaluationResult,
    NextVideoDatasetSpec,
    NextVideoEvaluationLimits,
    NextVideoEvaluationSettings,
    NextVideoPromptSettings,
    run_next_video_evaluation,
)
from .model import GenerationSettings, load_tokenizer_and_model
from .opinion import (
    OpinionDatasetSpec,
    OpinionEvaluationControls,
    OpinionEvaluationResult,
    OpinionEvaluationSettings,
    OpinionInferenceContext,
    OpinionPromptSettings,
    OpinionStudyFiles,
    OpinionStudyResult,
    OpinionStudySummary,
    run_opinion_evaluation,
)
from .reports import generate_reports

LOGGER = logging.getLogger("grpo.pipeline")


def _status(message: str, *args) -> None:
    """Log ``message`` at INFO level and mirror it to stdout immediately."""

    text = message % args if args else message
    LOGGER.info(text)
    print(f"[grpo.pipeline] {text}", flush=True)

DEFAULT_REGENERATE_HINT = (
    "Regenerate via `python -m grpo.pipeline --stage full` after producing "
    "updated evaluation artifacts under `models/grpo/`."
)


def configure_logging(log_level: str | int) -> None:
    """Configure structured logging and enable faulthandler early.

    :param log_level: Logging verbosity requested via CLI (string name or level).
    :returns: ``None``. The root logger is configured in-place.
    """

    if isinstance(log_level, str):
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    else:
        numeric_level = int(log_level)

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    try:  # pragma: no cover - faulthandler may already be active
        faulthandler.enable()
    except RuntimeError:
        LOGGER.debug("faulthandler already enabled")
    LOGGER.debug(
        "Logging configured at level %s", logging.getLevelName(numeric_level)
    )


def _safe_float(value: object) -> float | None:
    """Return ``value`` coerced to ``float`` when possible."""

    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if math.isnan(result):
        return None
    return result


def _safe_int(value: object) -> int | None:
    """Return ``value`` coerced to ``int`` when possible."""

    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fmt_rate(value: float | None, digits: int = 3) -> str:
    """Return a formatted rate or an em dash when ``value`` is missing."""

    return f"{value:.{digits}f}" if value is not None else "—"


def _fmt_count(value: int | None) -> str:
    """Return an integer with thousands separators or an em dash."""

    return f"{value:,}" if value is not None else "—"


def _extract_next_video_metrics(payload: Mapping[str, object] | None) -> Mapping[str, object]:
    """Normalise next-video metrics payloads that may wrap values under ``metrics``."""

    if not isinstance(payload, Mapping):
        return {}
    if "accuracy_overall" in payload:
        return payload
    nested = payload.get("metrics")
    if isinstance(nested, Mapping):
        return nested
    return payload


def _log_next_video_summary(result: NextVideoEvaluationResult | None) -> None:
    """Emit INFO-level summary metrics for ``result``."""

    if result is None:
        LOGGER.info("Next-video metrics unavailable; skipping summary log.")
        return

    metrics = _extract_next_video_metrics(result.metrics)
    if not metrics:
        LOGGER.info("Next-video metrics payload empty; skipping summary log.")
        return

    accuracy = _safe_float(metrics.get("accuracy_overall"))
    parsed_rate = _safe_float(metrics.get("parsed_rate"))
    format_rate = _safe_float(metrics.get("format_rate"))
    eligible = _safe_int(metrics.get("n_eligible"))
    total = _safe_int(metrics.get("n_total"))
    baseline_block = metrics.get("baseline_most_frequent_gold_index")
    baseline_accuracy = _safe_float(baseline_block.get("accuracy")) if isinstance(
        baseline_block, Mapping
    ) else None
    random_accuracy = _safe_float(metrics.get("random_baseline_expected_accuracy"))
    delta_accuracy = (
        accuracy - baseline_accuracy
        if accuracy is not None and baseline_accuracy is not None
        else None
    )

    LOGGER.info(
        "Next-video metrics | accuracy=%s | baseline=%s | Δ=%s | random=%s | parsed=%s | format=%s | eligible=%s/%s",
        _fmt_rate(accuracy),
        _fmt_rate(baseline_accuracy),
        _fmt_rate(delta_accuracy),
        _fmt_rate(random_accuracy),
        _fmt_rate(parsed_rate),
        _fmt_rate(format_rate),
        _fmt_count(eligible),
        _fmt_count(total),
    )

    if isinstance(baseline_block, Mapping):
        top_index = baseline_block.get("top_index")
        baseline_count = _safe_int(baseline_block.get("count"))
        LOGGER.debug(
            "Baseline (most frequent gold index): index=%s count=%s",
            top_index if top_index is not None else "—",
            _fmt_count(baseline_count),
        )


def _weighted_baseline_direction(result: OpinionEvaluationResult | None) -> float | None:
    """Return an eligible-weighted baseline directional accuracy for ``result``."""

    if result is None:
        return None
    numerator = 0.0
    denominator = 0
    for study in result.studies:
        baseline = getattr(study, "baseline", None)
        eligible = getattr(study, "eligible", None)
        baseline_direction = (
            _safe_float(baseline.get("direction_accuracy"))
            if isinstance(baseline, Mapping)
            else None
        )
        if baseline_direction is None or not eligible:
            continue
        denominator += int(eligible)
        numerator += baseline_direction * int(eligible)
    if denominator:
        return numerator / denominator
    return None


def _log_opinion_summary(result: OpinionEvaluationResult | None) -> None:
    """Emit INFO-level summary metrics for aggregated opinion results."""

    if result is None:
        LOGGER.info("Opinion metrics unavailable; skipping summary log.")
        return

    combined = result.combined_metrics if isinstance(result.combined_metrics, Mapping) else {}
    if not combined:
        LOGGER.info("Opinion combined metrics payload empty; skipping summary log.")
        return

    direction_accuracy = _safe_float(combined.get("direction_accuracy"))
    mae_after = _safe_float(combined.get("mae_after"))
    rmse_after = _safe_float(combined.get("rmse_after"))
    mae_change = _safe_float(combined.get("mae_change"))
    rmse_change = _safe_float(combined.get("rmse_change"))
    eligible = _safe_int(combined.get("eligible"))
    baseline_direction = _weighted_baseline_direction(result)
    delta_direction = (
        direction_accuracy - baseline_direction
        if direction_accuracy is not None and baseline_direction is not None
        else None
    )
    participants = sum(int(getattr(study, "participants", 0)) for study in result.studies)

    LOGGER.info(
        "Opinion metrics | direction=%s | baseline=%s | Δ=%s | mae_after=%s | rmse_after=%s | mae_change=%s | rmse_change=%s | eligible=%s | participants=%s",
        _fmt_rate(direction_accuracy),
        _fmt_rate(baseline_direction),
        _fmt_rate(delta_direction),
        _fmt_rate(mae_after),
        _fmt_rate(rmse_after),
        _fmt_rate(mae_change),
        _fmt_rate(rmse_change),
        _fmt_count(eligible),
        _fmt_count(participants),
    )


@dataclass(frozen=True)
class PipelineContext:
    """Filesystem locations reused across pipeline stages.

    :ivar Path repo_root: Absolute path to the repository root.
    :ivar Path out_dir: Base directory for all GRPO evaluation artifacts.
    :ivar Path next_video_root: Directory containing next-video evaluation runs.
    :ivar Path opinion_root: Directory containing opinion evaluation runs.
    :ivar str label: Human-readable label for the current evaluation run.
    """

    repo_root: Path
    out_dir: Path
    next_video_root: Path
    opinion_root: Path
    label: str

    @property
    def next_video_run_dir(self) -> Path:
        """Return the next-video run directory for the configured label.

        :returns: Directory where next-video artifacts for the run are stored.
        :rtype: Path
        """
        return self.next_video_root / self.label

    @property
    def opinion_run_dir(self) -> Path:
        """Return the opinion run directory for the configured label.

        :returns: Directory where opinion artifacts for the run are stored.
        :rtype: Path
        """
        return self.opinion_root / self.label


@dataclass(frozen=True)
class PipelinePrompts:
    """Resolved system prompts used across evaluation tasks.

    :ivar str system: System prompt forwarded to the model during evaluations.
    :ivar str opinion: System prompt specific to opinion evaluations.
    """

    system: str
    opinion: str


@dataclass
class PipelineResults:
    """Container for pipeline evaluation results.

    :ivar NextVideoEvaluationResult next_video: In-memory next-video metrics.
    :ivar OpinionEvaluationResult opinion: In-memory opinion metrics.
    """

    next_video: NextVideoEvaluationResult | None = None
    opinion: OpinionEvaluationResult | None = None


@dataclass(frozen=True)
class StageSelection:
    """Flags describing which pipeline stages should execute.

    :ivar str stage: Stage name selected via CLI.
    :ivar bool run_next_video: Whether to execute the next-video evaluation pass.
    :ivar bool run_opinion: Whether to execute the opinion evaluation pass.
    """

    stage: str
    run_next_video: bool
    run_opinion: bool

    @property
    def run_evaluations(self) -> bool:
        """Return ``True`` when evaluation stages should execute."""

        return self.stage in {"full", "evaluate"}

    @property
    def run_reports(self) -> bool:
        """Return ``True`` when report generation should execute."""

        return self.stage in {"full", "reports"}



def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse CLI arguments.

    :param argv: Optional sequence of CLI tokens supplied by the caller.
    :returns: Namespace containing all parsed CLI argument values.
    :rtype: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Evaluate GRPO checkpoints on next-video and opinion tasks."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="Dataset path or HF id.")
    parser.add_argument("--split", default=DEFAULT_EVAL_SPLIT, help="Evaluation split name.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Finetuned GRPO model path or hub identifier (required for evaluation stage).",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/tag when loading from the hub.",
    )
    parser.add_argument(
        "--dtype",
        default="auto",
        help="Torch dtype for model loading (e.g. bfloat16, float16, auto).",
    )
    parser.add_argument(
        "--label",
        default=None,
        help="Human-readable label for report directories (defaults to model name).",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Directory receiving evaluation artifacts (defaults to <repo>/models/grpo).",
    )
    parser.add_argument(
        "--reports-subdir",
        default="grpo",
        help="Subdirectory under reports/ to store Markdown summaries.",
    )
    parser.add_argument(
        "--baseline-label",
        default="GRPO",
        help="Display name for the baseline used in report headings.",
    )
    parser.add_argument(
        "--regenerate-hint",
        default=DEFAULT_REGENERATE_HINT,
        help=(
            "Sentence appended to the catalog README describing how to refresh artefacts. "
            "Pass an empty string to omit."
        ),
    )
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="Optional path containing the GRPO system prompt.",
    )
    parser.add_argument(
        "--opinion-prompt-file",
        default=None,
        help="Optional path containing the opinion evaluation system prompt.",
    )
    parser.add_argument(
        "--solution-key",
        default="next_video_id",
        help="Dataset column containing the gold next-video identifier.",
    )
    parser.add_argument(
        "--max-history",
        type=int,
        default=12,
        help="Maximum watch-history depth forwarded to prompt_builder.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature used during generation.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Optional nucleus sampling top-p parameter.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum number of tokens generated per completion.",
    )
    parser.add_argument(
        "--eval-max",
        type=int,
        default=0,
        help="Limit next-video evaluation rows (0 keeps all).",
    )
    parser.add_argument(
        "--issues",
        default="",
        help="Comma-separated issue filters for next-video evaluation.",
    )
    parser.add_argument(
        "--studies",
        default="",
        help="Comma-separated participant-study filters for next-video evaluation.",
    )
    parser.add_argument(
        "--opinion-studies",
        default="",
        help="Comma-separated opinion study keys to evaluate.",
    )
    parser.add_argument(
        "--opinion-max-participants",
        type=int,
        default=0,
        help="Optional cap on participants per opinion study.",
    )
    parser.add_argument(
        "--direction-tolerance",
        type=float,
        default=1e-6,
        help="Tolerance for treating opinion deltas as no-change.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing evaluation artifacts.",
    )
    parser.add_argument(
        "--no-next-video",
        action="store_true",
        help="Skip next-video evaluation and reporting.",
    )
    parser.add_argument(
        "--no-opinion",
        action="store_true",
        help="Skip opinion evaluation and reporting.",
    )
    parser.add_argument(
        "--stage",
        choices=["full", "evaluate", "reports"],
        default="full",
        help="Select which stage(s) to run.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, etc.).",
    )
    return parser.parse_args(argv)


def _load_prompt_from_file(path: str | None, *, fallback: str) -> str:
    """Return prompt text from ``path`` or ``fallback`` when path is ``None``.

    :param path: Optional filesystem path pointing to a prompt file.
    :param fallback: Default prompt string returned when no file is supplied.
    :returns: Raw prompt text read from disk or the fallback string.
    :rtype: str
    """
    if not path:
        return fallback
    prompt_path = Path(path)
    with prompt_path.open("r", encoding="utf-8") as handle:
        return handle.read()


def _comma_separated(raw: str) -> tuple[str, ...]:
    """Split comma-separated CLI tokens into a tuple.

    :param raw: Raw comma-delimited string from the CLI.
    :returns: Normalized tuple of stripped tokens; empty when ``raw`` is falsy.
    :rtype: tuple[str, ...]
    """
    if not raw:
        return ()
    return tuple(token.strip() for token in raw.split(",") if token.strip())


def _build_context(args: argparse.Namespace) -> PipelineContext:
    """Return shared filesystem paths derived from CLI arguments.

    :param args: Parsed argument namespace produced by :func:`_parse_args`.
    :returns: Dataclass describing canonical filesystem locations.
    :rtype: PipelineContext
    """
    out_dir = _resolve_out_dir(args)
    return PipelineContext(
        repo_root=_repo_root(),
        out_dir=out_dir,
        next_video_root=out_dir / "next_video",
        opinion_root=out_dir / "opinion",
        label=_derive_label(args),
    )


def _load_prompts(args: argparse.Namespace) -> PipelinePrompts:
    """Resolve system prompts from CLI arguments or fallbacks.

    :param args: Parsed CLI arguments.
    :returns: Dataclass containing system and opinion prompts.
    :rtype: PipelinePrompts
    """
    return PipelinePrompts(
        system=_load_prompt_from_file(args.system_prompt_file, fallback=DEFAULT_SYSTEM_PROMPT),
        opinion=_load_prompt_from_file(
            args.opinion_prompt_file,
            fallback=OPINION_SYSTEM_PROMPT,
        ),
    )


def _resolve_stage_selection(args: argparse.Namespace) -> StageSelection:
    """Summarize which pipeline components are requested.

    :param args: Parsed CLI arguments.
    :returns: Flags specifying which stages should run.
    :rtype: StageSelection
    """
    return StageSelection(
        stage=args.stage,
        run_next_video=not args.no_next_video,
        run_opinion=not args.no_opinion,
    )


def _derive_label(args: argparse.Namespace) -> str:
    """Return the directory label identifying this evaluation run.

    :param args: Parsed CLI arguments.
    :returns: Label used to name output directories for the run.
    :rtype: str
    """
    if args.label:
        return args.label.strip()
    if args.model:
        model_name = Path(args.model).name
        return model_name.replace("/", "_").replace(" ", "_")
    return "grpo"


def _resolve_out_dir(args: argparse.Namespace) -> Path:
    """Return the base output directory for evaluation artifacts.

    :param args: Parsed CLI arguments.
    :returns: Filesystem path serving as base for pipeline outputs.
    :rtype: Path
    """
    if args.out_dir:
        return Path(args.out_dir)
    return _repo_root() / "models" / "grpo"


def _load_json(path: Path) -> Mapping[str, object]:
    """Load JSON payload from ``path``.

    :param path: Path pointing to the JSON file to read.
    :returns: Parsed JSON mapping.
    :rtype: Mapping[str, object]
    """
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_next_video_from_disk(run_dir: Path) -> NextVideoEvaluationResult | None:
    """Return a :class:`NextVideoEvaluationResult` by reading existing metrics.

    :param run_dir: Directory containing cached next-video artifacts.
    :returns: Materialized evaluation result or ``None`` when metrics are missing.
    :rtype: NextVideoEvaluationResult | None
    """
    metrics_path = run_dir / "metrics.json"
    predictions_path = run_dir / "predictions.jsonl"
    qa_log_path = run_dir / "qa.log"
    if not metrics_path.exists():
        LOGGER.warning("Next-video metrics not found at %s; skipping report.", metrics_path)
        return None
    metrics = _load_json(metrics_path)
    return NextVideoEvaluationResult(
        run_dir=run_dir,
        metrics_path=metrics_path,
        predictions_path=predictions_path,
        qa_log_path=qa_log_path,
        metrics=metrics,
    )


def _resolve_opinion_spec(key: str):
    """Return the opinion spec matching ``key`` (case-insensitive).

    :param key: Study identifier supplied via the CLI.
    :returns: Matching opinion study specification or ``None`` when unknown.
    """
    lowered = key.lower()
    for spec in DEFAULT_SPECS:
        if spec.key in {key, lowered}:
            return spec
    return None


def _build_opinion_study(study_dir: Path) -> OpinionStudyResult | None:
    """Return an :class:`OpinionStudyResult` derived from disk caches.

    :param study_dir: Path to an individual opinion study directory.
    :returns: Study result populated from cached metrics, or ``None`` if incomplete.
    :rtype: OpinionStudyResult | None
    """
    if not study_dir.is_dir():
        return None
    metrics_path = study_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    payload = _load_json(metrics_path)
    metrics = payload.get("metrics", payload)
    spec = _resolve_opinion_spec(study_dir.name)
    if spec is None:
        LOGGER.warning("Unknown opinion study directory %s; skipping.", study_dir)
        return None
    files = OpinionStudyFiles(
        metrics=metrics_path,
        predictions=study_dir / "predictions.jsonl",
        qa_log=study_dir / "qa.log",
    )
    summary = OpinionStudySummary(
        metrics=metrics,
        baseline=payload.get("baseline", {}),
        participants=int(metrics.get("participants") or 0),
        eligible=int(metrics.get("eligible", 0)),
    )
    return OpinionStudyResult(
        study=spec,
        files=files,
        summary=summary,
    )


def _load_opinion_from_disk(out_dir: Path) -> OpinionEvaluationResult | None:
    """Return an :class:`OpinionEvaluationResult` using cached metrics.

    :param out_dir: Directory containing opinion evaluation artifacts.
    :returns: Materialized opinion metrics, or ``None`` if cache is missing.
    :rtype: OpinionEvaluationResult | None
    """
    combined_path = out_dir / "combined_metrics.json"
    if not combined_path.exists():
        LOGGER.warning("Opinion combined metrics missing at %s; skipping report.", combined_path)
        return None
    combined_payload = _load_json(combined_path)
    combined_metrics = combined_payload.get("metrics", combined_payload)

    studies: list[OpinionStudyResult] = []
    for study_dir in sorted(out_dir.glob("*")):
        study = _build_opinion_study(study_dir)
        if study is not None:
            studies.append(study)

    return OpinionEvaluationResult(studies=studies, combined_metrics=combined_metrics)


def _run_evaluations(
    args: argparse.Namespace,
    selection: StageSelection,
    context: PipelineContext,
    prompts: PipelinePrompts,
) -> PipelineResults:
    """Execute next-video and opinion evaluations when requested.

    :param args: Parsed CLI arguments.
    :param selection: Stage flags indicating which evaluations to run.
    :param context: Shared filesystem context.
    :param prompts: Resolved system prompts for evaluations.
    :returns: Dataclass containing evaluation results and cached metrics.
    :rtype: PipelineResults
    """
    _status(
        "Beginning evaluation run label=%s stage=%s next_video=%s opinion=%s",
        context.label,
        selection.stage,
        selection.run_next_video,
        selection.run_opinion,
    )
    results = PipelineResults()
    generation = GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    load_start = time.perf_counter()
    _status(
        "Loading tokenizer/model from %s (revision=%s, dtype=%s)",
        args.model,
        args.revision or "default",
        args.dtype,
    )
    try:
        tokenizer, model = load_tokenizer_and_model(
            args.model,
            revision=args.revision,
            dtype=args.dtype,
            trust_remote_code=True,
        )
    except Exception:  # pragma: no cover - surfaced for debugging
        LOGGER.exception("Model load failed for %s", args.model)
        raise
    load_elapsed = time.perf_counter() - load_start
    model_device = getattr(getattr(model, "device", None), "type", None) or getattr(
        model, "device", "unknown"
    )
    model_dtype = getattr(getattr(model, "dtype", None), "name", None) or getattr(
        model, "dtype", "unknown"
    )
    _status(
        "Model ready in %.2fs • tokenizer=%s model=%s device=%s dtype=%s",
        load_elapsed,
        type(tokenizer).__name__,
        type(model).__name__,
        model_device,
        model_dtype,
    )

    if selection.run_next_video:
        stage_start = time.perf_counter()
        _status("Running next-video evaluation for %s.", context.label)
        context.next_video_root.mkdir(parents=True, exist_ok=True)
        next_settings = NextVideoEvaluationSettings(
            model_label=context.label,
            dataset=NextVideoDatasetSpec(
                name=args.dataset,
                split=args.split,
                cache_dir=args.cache_dir,
            ),
            prompts=NextVideoPromptSettings(
                system_prompt=prompts.system,
                solution_key=args.solution_key,
                max_history=args.max_history,
            ),
            limits=NextVideoEvaluationLimits(max_examples=int(args.eval_max or 0)),
            overwrite=args.overwrite,
            generation=generation,
            filters=FilterSelection.from_raw(
                issues=_comma_separated(args.issues),
                studies=_comma_separated(args.studies),
            ),
        )
        results.next_video = run_next_video_evaluation(
            tokenizer=tokenizer,
            model=model,
            settings=next_settings,
            config_label=context.label,
            out_dir=context.next_video_root,
        )
        _status(
            "Next-video evaluation finished in %.2fs",
            time.perf_counter() - stage_start,
        )
        _log_next_video_summary(results.next_video)

    if selection.run_opinion:
        stage_start = time.perf_counter()
        _status("Running opinion evaluation.")
        opinion_settings = OpinionEvaluationSettings(
            dataset=OpinionDatasetSpec(
                name=args.dataset,
                split=args.split,
                cache_dir=args.cache_dir,
            ),
            prompts=OpinionPromptSettings(
                system=prompts.system,
                opinion=prompts.opinion,
                solution_key=args.solution_key,
                max_history=args.max_history,
            ),
            controls=OpinionEvaluationControls(
                max_participants=int(args.opinion_max_participants or 0),
                direction_tolerance=float(args.direction_tolerance),
                overwrite=bool(args.overwrite),
            ),
            include_studies=_comma_separated(args.opinion_studies),
        )
        opinion_dir = context.opinion_run_dir
        opinion_dir.mkdir(parents=True, exist_ok=True)
        results.opinion = run_opinion_evaluation(
            context=OpinionInferenceContext(
                tokenizer=tokenizer,
                model=model,
                generation=generation,
            ),
            settings=opinion_settings,
            out_dir=opinion_dir,
        )
        _status(
            "Opinion evaluation finished in %.2fs",
            time.perf_counter() - stage_start,
        )
        _log_opinion_summary(results.opinion)

    return results


def _generate_reports_if_needed(
    selection: StageSelection,
    context: PipelineContext,
    results: PipelineResults,
    args: argparse.Namespace,
) -> None:
    """Load cached results when necessary and generate reports.

    :param selection: Flags indicating which report types to render.
    :param context: Filesystem context describing where artifacts live.
    :param results: Evaluation results produced during the current invocation.
    :param args: CLI argument namespace providing reporting configuration.
    :returns: ``None``. Reports are materialized on disk.
    """
    next_result = results.next_video
    if selection.run_next_video and next_result is None:
        next_result = _load_next_video_from_disk(context.next_video_run_dir)
        if next_result is not None:
            LOGGER.info(
                "Loaded cached next-video metrics from %s", next_result.metrics_path
            )
            _log_next_video_summary(next_result)

    opinion_result = results.opinion
    if selection.run_opinion and opinion_result is None:
        opinion_result = _load_opinion_from_disk(context.opinion_run_dir)
        if opinion_result is not None:
            LOGGER.info(
                "Loaded cached opinion metrics from %s",
                context.opinion_run_dir / "combined_metrics.json",
            )
            _log_opinion_summary(opinion_result)

    generate_reports(
        repo_root=context.repo_root,
        next_video=next_result if selection.run_next_video else None,
        opinion=opinion_result if selection.run_opinion else None,
        options=ReportOptions(
            args.reports_subdir,  # reports_subdir
            args.baseline_label,  # baseline_label
            args.regenerate_hint or None,
        ),
    )
    _status(
        "Reports written under %s.",
        context.repo_root / "reports" / args.reports_subdir,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the GRPO evaluation pipeline.

    :param argv: Optional sequence of CLI arguments; defaults to ``sys.argv``.
    :returns: ``None``. Side effects include evaluation runs and report generation.
    """
    args = _parse_args(list(argv) if argv is not None else None)
    run_start = time.perf_counter()
    configure_logging(args.log_level)

    selection = _resolve_stage_selection(args)
    context = _build_context(args)
    prompts = _load_prompts(args)
    _status(
        "Stage selection: stage=%s run_evaluations=%s next_video=%s opinion=%s run_reports=%s",
        selection.stage,
        selection.run_evaluations,
        selection.run_next_video,
        selection.run_opinion,
        selection.run_reports,
    )
    _status(
        "Pipeline context: repo_root=%s out_dir=%s label=%s model=%s dataset=%s split=%s",
        context.repo_root,
        context.out_dir,
        context.label,
        args.model or "<unset>",
        args.dataset,
        args.split,
    )

    if selection.run_evaluations and not args.model:
        raise SystemExit("--model must be provided when running the evaluate stage.")

    results = PipelineResults()
    if selection.run_evaluations:
        results = _run_evaluations(args, selection, context, prompts)

    if selection.run_reports:
        _generate_reports_if_needed(selection, context, results, args)

    _status("Pipeline finished in %.2fs", time.perf_counter() - run_start)


if __name__ == "__main__":  # pragma: no cover
    main()

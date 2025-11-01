#!/usr/bin/env python
"""Pipeline runtime context for the KNN baselines.

Split from ``context.py`` to keep that module focused and concise.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Tuple


@dataclass(frozen=True)
class _OpinionPaths:
    """Grouped opinion-specific directories.

    :param dir: Base directory for opinion artifacts.
    :param sweep_dir: Directory used for opinion sweep intermediates.
    :param word2vec_dir: Directory for opinion Word2Vec models.
    """

    dir: Path
    sweep_dir: Path
    word2vec_dir: Path


@dataclass(frozen=True)
class _PipelinePaths:
    """Filesystem locations used by the pipeline.

    Grouping these reduces attribute count on :class:`~knn.pipeline.context.PipelineContext` while
    still exposing fine-grained read-only accessors.

    :param out_dir: Base output directory for the run.
    :param cache_dir: Cache directory for external datasets or models.
    :param sweep_dir: Directory for next-video sweep intermediates.
    :param word2vec_model_dir: Directory containing global Word2Vec models.
    :param next_video_dir: Base directory for next-video artifacts.
    :param opinion: Grouped opinion-specific directories.
    """

    out_dir: Path
    cache_dir: str
    sweep_dir: Path
    word2vec_model_dir: Path
    next_video_dir: Path
    opinion: _OpinionPaths


@dataclass(frozen=True)
class _ModelDefaults:
    """Default model hyper-parameters used by the pipeline.

    :param word2vec_epochs: Default number of Word2Vec training epochs.
    :param word2vec_workers: Thread workers for Word2Vec training.
    :param sentence_model: SentenceTransformer model identifier.
    :param sentence_device: Optional device string (e.g., ``"cuda"`` or ``"cpu"``).
    :param sentence_batch_size: Batch size for SentenceTransformer encoding.
    :param sentence_normalize: Whether to L2-normalize sentence embeddings.
    """

    word2vec_epochs: int
    word2vec_workers: int
    sentence_model: str
    sentence_device: str | None
    sentence_batch_size: int
    sentence_normalize: bool


@dataclass(frozen=True)
class _Workflow:
    """Execution toggles and resource options for the pipeline.

    :param feature_spaces: Feature spaces to evaluate (e.g., ``("tfidf", ...)``).
    :param jobs: Parallelism for CPU-bound steps.
    :param reuse_sweeps: Reuse existing sweep outputs if present.
    :param reuse_final: Reuse final evaluation results if present.
    :param allow_incomplete: Allow partial/cached results in reports.
    :param run_next_video: Whether to run next-video tasks.
    :param run_opinion: Whether to run opinion tasks.
    """

    feature_spaces: Tuple[str, ...]
    jobs: int
    reuse_sweeps: bool
    reuse_final: bool
    allow_incomplete: bool
    run_next_video: bool
    run_opinion: bool


@dataclass(frozen=True, init=False)
class PipelineContext:
    """Normalised configuration for a KNN pipeline run.

    Provides read-only accessors for most fields, while allowing controlled
    updates to sentence-transformer settings so cached artefacts can be reused
    across stages (see ``_align_sentence_transformer_context``).

    Parameters are accepted as keyword-only arguments to mirror the CLI layer
    and configuration loaders in the repository.

    :param dataset: Dataset identifier (e.g., ``"slate"``).
    :param out_dir: Base output directory for all artifacts.
    :param cache_dir: External cache directory path or name.
    :param sweep_dir: Directory for next-video sweep results.
    :param word2vec_model_dir: Directory containing Word2Vec models.
    :param next_video_dir: Directory for next-video artifacts.
    :param opinion_dir: Directory for opinion artifacts.
    :param opinion_sweep_dir: Directory for opinion sweep results.
    :param opinion_word2vec_dir: Directory containing opinion Word2Vec models.
    :param k_sweep: K-values to sweep over, encoded as a string.
    :param study_tokens: Optional study filters limiting evaluation.
    :param word2vec_epochs: Default Word2Vec epochs.
    :param word2vec_workers: Default Word2Vec workers.
    :param sentence_model: SentenceTransformer model identifier.
    :param sentence_device: Optional device string for SentenceTransformer.
    :param sentence_batch_size: Batch size for SentenceTransformer encoding.
    :param sentence_normalize: Whether to L2-normalize sentence embeddings.
    :param feature_spaces: Feature spaces to include in runs.
    :param jobs: Parallel CPU worker count.
    :param reuse_sweeps: Reuse existing sweep outputs when available.
    :param reuse_final: Reuse existing final evaluation outputs.
    :param allow_incomplete: Allow partially cached runs to be reported.
    :param run_next_video: Enable next-video tasks.
    :param run_opinion: Enable opinion tasks.
    :returns: Constructed immutable context instance.
    """

    # Simple public fields
    dataset: str
    k_sweep: str
    study_tokens: Tuple[str, ...]

    # Grouped private bundles
    _paths: _PipelinePaths
    _models: _ModelDefaults
    _workflow: _Workflow

    def __init__(
        self,
        *,
        # Common flat fields always required
        dataset: str,
        # Either grouped bundles OR flat kwargs provided in **flat
        k_sweep: str | None = None,
        study_tokens: Tuple[str, ...] | None = None,
        paths: _PipelinePaths | None = None,
        models: _ModelDefaults | None = None,
        workflow: _Workflow | None = None,
        **flat: object,
    ) -> None:
        """Construct a pipeline context.

        Accepts grouped bundles (``paths``, ``models``, ``workflow``) or a set
        of flat keyword arguments for backwards compatibility with existing
        tests and call-sites.
        """

        # Resolve grouped bundles if not provided
        if paths is None:
            # Validate required legacy fields
            assert flat.get("out_dir") is not None
            assert flat.get("cache_dir") is not None
            assert flat.get("sweep_dir") is not None
            assert flat.get("word2vec_model_dir") is not None
            assert flat.get("next_video_dir") is not None
            assert flat.get("opinion_dir") is not None
            assert flat.get("opinion_sweep_dir") is not None
            assert flat.get("opinion_word2vec_dir") is not None
            paths = _PipelinePaths(
                out_dir=flat["out_dir"],  # type: ignore[index]
                cache_dir=str(flat["cache_dir"]),  # type: ignore[index]
                sweep_dir=flat["sweep_dir"],  # type: ignore[index]
                word2vec_model_dir=flat["word2vec_model_dir"],  # type: ignore[index]
                next_video_dir=flat["next_video_dir"],  # type: ignore[index]
                opinion=_OpinionPaths(
                    dir=flat["opinion_dir"],  # type: ignore[index]
                    sweep_dir=flat["opinion_sweep_dir"],  # type: ignore[index]
                    word2vec_dir=flat["opinion_word2vec_dir"],  # type: ignore[index]
                ),
            )

        if models is None:
            assert flat.get("word2vec_epochs") is not None
            assert flat.get("word2vec_workers") is not None
            assert flat.get("sentence_model") is not None
            assert flat.get("sentence_batch_size") is not None
            assert flat.get("sentence_normalize") is not None
            models = _ModelDefaults(
                word2vec_epochs=int(flat["word2vec_epochs"]),  # type: ignore[index]
                word2vec_workers=int(flat["word2vec_workers"]),  # type: ignore[index]
                sentence_model=str(flat["sentence_model"]),  # type: ignore[index]
                sentence_device=flat.get("sentence_device"),
                sentence_batch_size=int(flat["sentence_batch_size"]),  # type: ignore[index]
                sentence_normalize=bool(flat["sentence_normalize"]),  # type: ignore[index]
            )

        if workflow is None:
            assert flat.get("feature_spaces") is not None
            assert flat.get("jobs") is not None
            workflow = _Workflow(
                feature_spaces=tuple(flat["feature_spaces"]),  # type: ignore[index]
                jobs=int(flat["jobs"]),  # type: ignore[index]
                reuse_sweeps=(
                    bool(flat.get("reuse_sweeps"))
                    if flat.get("reuse_sweeps") is not None
                    else False
                ),
                reuse_final=(
                    bool(flat.get("reuse_final")) if flat.get("reuse_final") is not None else False
                ),
                allow_incomplete=(
                    bool(flat.get("allow_incomplete"))
                    if flat.get("allow_incomplete") is not None
                    else False
                ),
                run_next_video=(
                    bool(flat.get("run_next_video"))
                    if flat.get("run_next_video") is not None
                    else True
                ),
                run_opinion=(
                    bool(flat.get("run_opinion")) if flat.get("run_opinion") is not None else True
                ),
            )

        # Public simple fields
        object.__setattr__(self, "dataset", dataset)
        object.__setattr__(self, "k_sweep", str(k_sweep) if k_sweep is not None else "")
        object.__setattr__(self, "study_tokens", tuple(study_tokens or ()))

        # Grouped structures
        object.__setattr__(self, "_paths", paths)
        object.__setattr__(self, "_models", models)
        object.__setattr__(self, "_workflow", workflow)

    @classmethod
    def from_mappings(
        cls,
        *,
        paths: Mapping[str, object],
        settings: Mapping[str, object],
    ) -> "PipelineContext":
        """Factory building a context from resolved dictionaries.

        This keeps the ``__init__`` narrow while preserving the existing CLI
        call-sites that assemble simple ``dict`` objects.
        """
        path_bundle = _PipelinePaths(
            out_dir=paths["out_dir"],
            cache_dir=str(paths["cache_dir"]),
            sweep_dir=paths["sweep_dir"],
            word2vec_model_dir=paths["word2vec_model_dir"],
            next_video_dir=paths["next_video_dir"],
            opinion=_OpinionPaths(
                dir=paths["opinion_dir"],
                sweep_dir=paths["opinion_sweep_dir"],
                word2vec_dir=paths["opinion_word2vec_dir"],
            ),
        )
        model_bundle = _ModelDefaults(
            word2vec_epochs=int(settings["word2vec_epochs"]),
            word2vec_workers=int(settings["word2vec_workers"]),
            sentence_model=str(settings["sentence_model"]),
            sentence_device=settings.get("sentence_device"),
            sentence_batch_size=int(settings["sentence_batch_size"]),
            sentence_normalize=bool(settings["sentence_normalize"]),
        )
        workflow_bundle = _Workflow(
            feature_spaces=tuple(settings.get("feature_spaces") or ()),
            jobs=int(settings["jobs"]),
            reuse_sweeps=bool(settings["reuse_sweeps"]),
            reuse_final=bool(settings["reuse_final"]),
            allow_incomplete=bool(settings["allow_incomplete"]),
            run_next_video=bool(settings["run_next_video"]),
            run_opinion=bool(settings["run_opinion"]),
        )

        return cls(
            dataset=str(paths["dataset"]),
            k_sweep=str(settings["k_sweep"]),
            study_tokens=tuple(settings.get("study_tokens") or ()),
            paths=path_bundle,
            models=model_bundle,
            workflow=workflow_bundle,
        )

    def __getattr__(self, name: str):  # pragma: no cover - attribute forwarding
        """Provide dynamic access to grouped fields to keep API stable.

        Forwards path, model, and workflow attributes to their respective
        grouped dataclasses. This reduces the number of explicit public
        methods while preserving backwards-compatible attribute access.
        """
        # Path forwarding
        if name in {"out_dir", "cache_dir", "sweep_dir", "word2vec_model_dir", "next_video_dir"}:
            return getattr(self._paths, name)
        if name in {"opinion_dir", "opinion_sweep_dir", "opinion_word2vec_dir"}:
            mapping = {
                "opinion_dir": "dir",
                "opinion_sweep_dir": "sweep_dir",
                "opinion_word2vec_dir": "word2vec_dir",
            }
            return getattr(self._paths.opinion, mapping[name])
        # Model forwarding
        if name in {
            "word2vec_epochs",
            "word2vec_workers",
            "sentence_model",
            "sentence_device",
            "sentence_batch_size",
            "sentence_normalize",
        }:
            return getattr(self._models, name)
        # Workflow forwarding
        if name in {
            "feature_spaces",
            "jobs",
            "reuse_sweeps",
            "reuse_final",
            "allow_incomplete",
            "run_next_video",
            "run_opinion",
        }:
            return getattr(self._workflow, name)
        raise AttributeError(f"{type(self).__name__} has no attribute {name!r}")

    @property
    def sentence_device(self) -> str | None:  # pragma: no cover - simple forwarding
        """Device string used by SentenceTransformer.

        :returns: Device identifier such as ``"cuda"``, ``"cpu"``, or ``None``.
        :rtype: str | None
        """
        return self._models.sentence_device

    @sentence_device.setter
    def sentence_device(self, value: str | None) -> None:
        """Update the SentenceTransformer device while preserving other defaults.

        :param value: Device identifier (e.g., ``"cuda"``) or ``None``.
        :returns: None
        """
        models = self._models
        object.__setattr__(
            self,
            "_models",
            _ModelDefaults(
                word2vec_epochs=models.word2vec_epochs,
                word2vec_workers=models.word2vec_workers,
                sentence_model=models.sentence_model,
                sentence_device=value,
                sentence_batch_size=models.sentence_batch_size,
                sentence_normalize=models.sentence_normalize,
            ),
        )

    @property
    def sentence_batch_size(self) -> int:  # pragma: no cover - simple forwarding
        """Batch size for SentenceTransformer encoding.

        :returns: Number of examples processed per batch.
        :rtype: int
        """
        return self._models.sentence_batch_size

    @sentence_batch_size.setter
    def sentence_batch_size(self, value: int) -> None:
        """Update the SentenceTransformer batch size.

        :param value: New batch size.
        :returns: None
        """
        models = self._models
        object.__setattr__(
            self,
            "_models",
            _ModelDefaults(
                word2vec_epochs=models.word2vec_epochs,
                word2vec_workers=models.word2vec_workers,
                sentence_model=models.sentence_model,
                sentence_device=models.sentence_device,
                sentence_batch_size=int(value),
                sentence_normalize=models.sentence_normalize,
            ),
        )

    @property
    def sentence_normalize(self) -> bool:  # pragma: no cover - simple forwarding
        """Whether SentenceTransformer embeddings are L2-normalized.

        :returns: ``True`` when normalization is enabled.
        :rtype: bool
        """
        return self._models.sentence_normalize

    @sentence_normalize.setter
    def sentence_normalize(self, value: bool) -> None:
        """Toggle L2-normalization of SentenceTransformer embeddings.

        :param value: Normalization flag.
        :returns: None
        """
        models = self._models
        object.__setattr__(
            self,
            "_models",
            _ModelDefaults(
                word2vec_epochs=models.word2vec_epochs,
                word2vec_workers=models.word2vec_workers,
                sentence_model=models.sentence_model,
                sentence_device=models.sentence_device,
                sentence_batch_size=models.sentence_batch_size,
                sentence_normalize=bool(value),
            ),
        )

    # Keep only setters as explicit methods; getters are forwarded via __getattr__

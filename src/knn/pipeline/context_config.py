#!/usr/bin/env python
"""Sweep configuration types for the KNN pipeline.

Split from ``context.py`` to keep that module focused and concise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class _Word2VecConfig:
    """Grouped Word2Vec hyper-parameters for a sweep config.

    :param size: Vector dimensionality.
    :param window: Context window size.
    :param min_count: Minimum token frequency to include.
    :param epochs: Number of training epochs.
    :param workers: Worker processes for training.
    """

    size: int | None = None
    window: int | None = None
    min_count: int | None = None
    epochs: int | None = None
    workers: int | None = None


@dataclass(frozen=True)
class _SentenceTransformerConfig:
    """Grouped SentenceTransformer parameters for a sweep config.

    :param model: Model identifier (e.g. 'all-MiniLM-L6-v2').
    :param device: Target device string (e.g. 'cpu', 'cuda').
    :param batch_size: Batch size for encoding.
    :param normalize: Whether to L2-normalize sentence embeddings.
    """

    model: str | None = None
    device: str | None = None
    batch_size: int | None = None
    normalize: bool | None = None


@dataclass(frozen=True, init=False)
class SweepConfig:
    """Describe a single hyper-parameter configuration scheduled for execution.

    :param feature_space: One of 'tfidf', 'word2vec', or 'sentence_transformer'.
    :param metric: Target metric label used for selection.
    :param text_fields: Ordered text fields to include when building feature space.
    :param word2vec_size: Word2Vec vector size.
    :param word2vec_window: Word2Vec context window.
    :param word2vec_min_count: Word2Vec minimum token count.
    :param word2vec_epochs: Word2Vec training epochs.
    :param word2vec_workers: Word2Vec worker processes.
    :param sentence_transformer_model: SentenceTransformer model name.
    :param sentence_transformer_device: Device hint for encoding.
    :param sentence_transformer_batch_size: Batch size for encoding.
    :param sentence_transformer_normalize: Whether to L2-normalize embeddings.
    """

    feature_space: str
    metric: str
    text_fields: Tuple[str, ...]
    _w2v: _Word2VecConfig | None = None
    _sentence: _SentenceTransformerConfig | None = None

    def __init__(
        self,
        *,
        feature_space: str,
        metric: str,
        text_fields: Tuple[str, ...],
        word2vec_size: int | None = None,
        word2vec_window: int | None = None,
        word2vec_min_count: int | None = None,
        word2vec_epochs: int | None = None,
        word2vec_workers: int | None = None,
        sentence_transformer_model: str | None = None,
        sentence_transformer_device: str | None = None,
        sentence_transformer_batch_size: int | None = None,
        sentence_transformer_normalize: bool | None = None,
    ) -> None:
        object.__setattr__(self, "feature_space", feature_space)
        object.__setattr__(self, "metric", metric)
        object.__setattr__(self, "text_fields", tuple(text_fields))
        w2v = None
        if any(
            v is not None
            for v in (
                word2vec_size,
                word2vec_window,
                word2vec_min_count,
                word2vec_epochs,
                word2vec_workers,
            )
        ):
            w2v = _Word2VecConfig(
                size=word2vec_size,
                window=word2vec_window,
                min_count=word2vec_min_count,
                epochs=word2vec_epochs,
                workers=word2vec_workers,
            )
        object.__setattr__(self, "_w2v", w2v)
        sentence = None
        if any(
            v is not None
            for v in (
                sentence_transformer_model,
                sentence_transformer_device,
                sentence_transformer_batch_size,
                sentence_transformer_normalize,
            )
        ):
            sentence = _SentenceTransformerConfig(
                model=sentence_transformer_model,
                device=sentence_transformer_device,
                batch_size=sentence_transformer_batch_size,
                normalize=sentence_transformer_normalize,
            )
        object.__setattr__(self, "_sentence", sentence)

    @property
    def word2vec_size(self) -> int | None:  # pragma: no cover - simple forwarding
        """Word2Vec vector dimensionality for this configuration.

        :returns: Vector size override or ``None`` if unspecified.
        :rtype: int | None
        """
        return self._w2v.size if self._w2v is not None else None

    @property
    def word2vec_window(self) -> int | None:  # pragma: no cover - simple forwarding
        """Word2Vec context window size for this configuration.

        :returns: Window size override or ``None`` if unspecified.
        :rtype: int | None
        """
        return self._w2v.window if self._w2v is not None else None

    @property
    def word2vec_min_count(self) -> int | None:  # pragma: no cover - simple forwarding
        """Minimum token frequency threshold used by Word2Vec.

        :returns: Minimum count override or ``None`` if unspecified.
        :rtype: int | None
        """
        return self._w2v.min_count if self._w2v is not None else None

    @property
    def word2vec_epochs(self) -> int | None:  # pragma: no cover - simple forwarding
        """Number of training epochs for Word2Vec.

        :returns: Epoch count override or ``None`` if unspecified.
        :rtype: int | None
        """
        return self._w2v.epochs if self._w2v is not None else None

    @property
    def word2vec_workers(self) -> int | None:  # pragma: no cover - simple forwarding
        """Number of worker processes used for Word2Vec training.

        :returns: Worker count override or ``None`` if unspecified.
        :rtype: int | None
        """
        return self._w2v.workers if self._w2v is not None else None

    @property
    def sentence_transformer_model(self) -> str | None:  # pragma: no cover - simple forwarding
        """SentenceTransformer model identifier to use for encoding.

        :returns: Model name override or ``None`` if unspecified.
        :rtype: str | None
        """
        return self._sentence.model if self._sentence is not None else None

    @property
    def sentence_transformer_device(self) -> str | None:  # pragma: no cover - simple forwarding
        """Target device used for SentenceTransformer encoding.

        :returns: Device override (e.g. ``"cpu"`` or ``"cuda"``) or ``None``.
        :rtype: str | None
        """
        return self._sentence.device if self._sentence is not None else None

    @property
    def sentence_transformer_batch_size(self) -> int | None:  # pragma: no cover - simple forwarding
        """Batch size used for SentenceTransformer encoding.

        :returns: Batch size override or ``None`` if unspecified.
        :rtype: int | None
        """
        return self._sentence.batch_size if self._sentence is not None else None

    @property
    def sentence_transformer_normalize(self) -> bool | None:  # pragma: no cover - simple forwarding
        """Whether to L2-normalize SentenceTransformer embeddings.

        :returns: Normalization flag override or ``None`` if unspecified.
        :rtype: bool | None
        """
        return self._sentence.normalize if self._sentence is not None else None

    def label(self) -> str:
        """Create a filesystem-friendly identifier summarising the configuration.

        :returns: A compact, safe string encoding of key parameters.
        """
        text_label = "none"
        if self.text_fields:
            text_label = "_".join(field.replace("_", "") for field in self.text_fields)

        def _sanitize_token(value: str) -> str:
            """Collapse problematic characters to keep tokens filesystem friendly.

            :param value: Input token text.
            :returns: Alphanumeric-only token.
            """
            return "".join(char for char in value if char.isalnum())

        parts = [f"metric-{self.metric}", f"text-{text_label}"]
        if self.feature_space == "word2vec":
            if self.word2vec_size is not None:
                parts.append(f"size{int(self.word2vec_size)}")
            if self.word2vec_window is not None:
                parts.append(f"win{int(self.word2vec_window)}")
            if self.word2vec_min_count is not None:
                parts.append(f"minc{int(self.word2vec_min_count)}")
            if self.word2vec_epochs is not None:
                parts.append(f"ep{int(self.word2vec_epochs)}")
            if self.word2vec_workers is not None:
                parts.append(f"workers{int(self.word2vec_workers)}")
        elif self.feature_space == "sentence_transformer":
            model = self.sentence_transformer_model or ""
            if model:
                parts.append(_sanitize_token(model))
            if self.sentence_transformer_device:
                parts.append(f"device-{_sanitize_token(self.sentence_transformer_device)}")
            if self.sentence_transformer_batch_size is not None:
                parts.append(f"bs{int(self.sentence_transformer_batch_size)}")
            if self.sentence_transformer_normalize is not None:
                parts.append("norm" if self.sentence_transformer_normalize else "nonorm")
        return "_".join(parts)

    def cli_args(self, *, word2vec_model_dir: object | None = None) -> list[str]:
        """Serialise the configuration into CLI flags for knn.cli.main.

        :param word2vec_model_dir: Optional directory for Word2Vec caches, forwarded
            when the feature space is ``word2vec``.
        :returns: Argument vector encoding this configuration.
        """
        args: list[str] = [
            "--feature-space",
            self.feature_space,
            "--knn-metric",
            self.metric,
            "--knn-text-fields",
            ",".join(self.text_fields),
        ]
        if self.feature_space == "word2vec":
            if self.word2vec_size is not None:
                args.extend(["--word2vec-size", str(int(self.word2vec_size))])
            if self.word2vec_window is not None:
                args.extend(["--word2vec-window", str(int(self.word2vec_window))])
            if self.word2vec_min_count is not None:
                args.extend(["--word2vec-min-count", str(int(self.word2vec_min_count))])
            if self.word2vec_epochs is not None:
                args.extend(["--word2vec-epochs", str(int(self.word2vec_epochs))])
            if self.word2vec_workers is not None:
                args.extend(["--word2vec-workers", str(int(self.word2vec_workers))])
            if word2vec_model_dir is not None:
                args.extend(["--word2vec-model-dir", str(word2vec_model_dir)])
        elif self.feature_space == "sentence_transformer":
            if self.sentence_transformer_model is not None:
                args.extend(["--sentence-transformer-model", self.sentence_transformer_model])
            if self.sentence_transformer_device:
                args.extend(["--sentence-transformer-device", self.sentence_transformer_device])
            if self.sentence_transformer_batch_size is not None:
                args.extend([
                    "--sentence-transformer-batch-size",
                    str(int(self.sentence_transformer_batch_size)),
                ])
            if self.sentence_transformer_normalize is not None:
                args.append(
                    "--sentence-transformer-normalize"
                    if self.sentence_transformer_normalize
                    else "--sentence-transformer-no-normalize"
                )
        return args

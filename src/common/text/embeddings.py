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

"""SentenceTransformer configuration and encoding helpers used in pipelines."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

try:  # pragma: no cover - optional dependency
    import torch  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[assignment]

__all__ = [
    "SentenceTransformerConfig",
    "SentenceTransformerEncoder",
    "sentence_transformer_config_from_args",
]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentenceTransformerConfig:
    """Runtime options controlling SentenceTransformer inference."""
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str | None = None
    batch_size: int = 32
    normalize: bool = True


def sentence_transformer_config_from_args(args: Any) -> SentenceTransformerConfig:
    """
    Build a :class:`SentenceTransformerConfig` from CLI-style arguments.

    :param args: Object exposing attributes for the relevant CLI options.
    :returns: Populated SentenceTransformer configuration.
    """

    defaults = SentenceTransformerConfig()
    device = getattr(args, "sentence_transformer_device", "") or None
    return SentenceTransformerConfig(
        model_name=getattr(args, "sentence_transformer_model", defaults.model_name),
        device=device,
        batch_size=int(getattr(args, "sentence_transformer_batch_size", defaults.batch_size)),
        normalize=bool(getattr(args, "sentence_transformer_normalize", defaults.normalize)),
    )


class SentenceTransformerEncoder:
    """
    Thin wrapper over ``sentence_transformers`` with lazy loading.

    :ivar config: Runtime configuration controlling inference behaviour.
    :vartype config: SentenceTransformerConfig
    :ivar _model: Lazily initialised SentenceTransformer instance (``None`` until used).
    :vartype _model: SentenceTransformer | None
    """


    def __init__(self, config: SentenceTransformerConfig) -> None:
        """
        Initialise the encoder with runtime configuration.

        :param config: Behavioural options used when loading and invoking the model.
        :type config: SentenceTransformerConfig
        """


        self.config = config
        self._model: SentenceTransformer | None = None  # type: ignore[valid-type]

    @staticmethod
    def _should_retry_on_cpu(device: str | None, error: Exception) -> bool:
        """
        Return ``True`` when a CUDA-capable device string should be retried on CPU.

        :param device: Device identifier requested for SentenceTransformer inference.
        :type device: str | None
        :param error: Exception raised during model initialisation on ``device``.
        :type error: Exception
        :returns: ``True`` when the error suggests retrying on CPU, ``False`` otherwise.
        :rtype: bool
        """

        if not device:
            return False

        prefix = device.split(":", 1)[0].lower()
        message = str(error).lower()
        if prefix not in {"cuda", "gpu"} or not any(
            token in message for token in ("cuda", "gpu", "nvidia")
        ):
            return False

        should_retry = True
        if torch is not None:
            cuda = getattr(torch, "cuda", None)
            if cuda is not None:
                is_available = getattr(cuda, "is_available", None)
                if callable(is_available):
                    try:
                        should_retry = not bool(is_available())
                    except (
                        AssertionError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ):
                        # pragma: no cover - best-effort detection
                        should_retry = True
        return should_retry

    def _ensure_model(self) -> SentenceTransformer:
        """
        Return a loaded SentenceTransformer model, initialising it when necessary.

        :returns: Cached SentenceTransformer instance ready for inference.
        :rtype: SentenceTransformer
        :raises ImportError: If the ``sentence-transformers`` package is unavailable.
        """


        if SentenceTransformer is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "Install sentence-transformers to use the sentence_transformer feature space."
            )
        if self._model is None:
            device = self.config.device
            try:
                self._model = SentenceTransformer(self.config.model_name, device=device)
            except RuntimeError as error:
                if self._should_retry_on_cpu(device, error):
                    LOGGER.warning(
                        "SentenceTransformer device '%s' unavailable; falling back to CPU.",
                        device,
                    )
                    self._model = SentenceTransformer(self.config.model_name, device="cpu")
                else:
                    raise
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a batch of texts into dense embeddings.

        :param texts: Sequence of documents to embed.
        :type texts: Sequence[str]
        :returns: Matrix of shape ``(len(texts), embedding_dimension)``.
        :rtype: numpy.ndarray
        """


        model = self._ensure_model()
        if not texts:
            dim = int(getattr(model, "get_sentence_embedding_dimension", lambda: 0)() or 0)
            return np.zeros((0, dim), dtype=np.float32)
        embeddings = model.encode(
            list(texts),
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
        )
        array = np.asarray(embeddings, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array

    def embedding_dimension(self) -> int:
        """
        Return the dimensionality of the produced embeddings.

        :returns: Number of features emitted by the underlying encoder.
        :rtype: int
        """


        model = self._ensure_model()
        if hasattr(model, "get_sentence_embedding_dimension"):
            return int(model.get_sentence_embedding_dimension())
        sample = self.encode(["sample"])
        return int(sample.shape[1] if sample.ndim == 2 else 0)

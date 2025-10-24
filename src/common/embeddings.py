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

"""Top-level orchestration helpers for the ``clean_data`` package.

This module stitches together the key pieces of the cleaning pipeline:
loading raw CodeOcean or Hugging Face datasets, filtering unusable rows,
converting interactions into prompt-ready examples, validating schema
requirements, saving artifacts, and dispatching prompt statistics reports.
It is the public surface that downstream tooling should import when they
need to build or persist cleaned prompt datasets. All functionality here is
distributed under the repository's Apache 2.0 license; see LICENSE for
details.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[assignment]

__all__ = ["SentenceTransformerConfig", "SentenceTransformerEncoder"]


@dataclass(frozen=True)
class SentenceTransformerConfig:
    """

    Configuration controlling SentenceTransformer inference.



    :ivar model_name: Attribute ``model_name``.

    :vartype model_name: str

    :ivar device: Attribute ``device``.

    :vartype device: str | None

    :ivar batch_size: Attribute ``batch_size``.

    :vartype batch_size: int

    :ivar normalize: Attribute ``normalize``.

    :vartype normalize: bool

    """


    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str | None = None
    batch_size: int = 32
    normalize: bool = True


class SentenceTransformerEncoder:
    """
    Thin wrapper over ``sentence_transformers`` with lazy loading.

    :ivar config: Runtime configuration controlling inference behavior.
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
            self._model = SentenceTransformer(self.config.model_name, device=self.config.device)
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

    def encode_iter(self, texts: Iterable[str]) -> np.ndarray:
        """
        Encode an iterable of texts by materialising it once.

        :param texts: Iterable of documents to embed.
        :type texts: Iterable[str]
        :returns: Matrix of embeddings corresponding to ``texts``.
        :rtype: numpy.ndarray
        """


        return self.encode(list(texts))

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

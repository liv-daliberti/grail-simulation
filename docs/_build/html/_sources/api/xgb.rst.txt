XGBoost Baseline API
====================

Vectoriser Options
------------------

The slate baseline supports multiple text featurisation strategies via the
``--text_vectorizer`` CLI switch (and the matching fields on
``XGBoostTrainConfig``):

* ``tfidf`` – sparse TF-IDF features (default, no extra dependencies).
* ``word2vec`` – averaged Word2Vec embeddings (requires ``gensim``).
* ``sentence_transformer`` – pooled Sentence-Transformer embeddings (requires
  the ``sentence-transformers`` package and a compatible PyTorch backend).

Use the accompanying CLI flags (``--word2vec-*`` or ``--sentence-transformer-*``)
to tune embedding dimensionality, model checkpoints, batch sizes, and
normalisation.

.. automodule:: xgb.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.vectorizers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.cli
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.utils
   :members:
   :undoc-members:
   :show-inheritance:

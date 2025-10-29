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
   :exclude-members: CandidateMetadata

.. automodule:: xgb.model
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.vectorizers
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SentenceTransformerConfig

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

Pipeline Orchestration
----------------------

The orchestration layer exposes dedicated stages for the slate-ranking and
opinion-regression tasks. Pass ``--tasks`` to ``python -m xgb.pipeline`` (or the
training wrapper scripts) to select which portions to run:

* ``next_video`` – plan and execute hyper-parameter sweeps for the slate model,
  promote the best configurations, and run the final validation pass.
* ``opinion`` – run the opinion-regression sweeps independently, select the best
  booster parameters, and train the regression models used in downstream
  reporting.

When a task is disabled the generated Markdown reports now include a short note
explaining that the section was intentionally skipped.

.. automodule:: xgb.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: SweepRunContext, OpinionStageConfig, OpinionSweepRunContext

.. automodule:: xgb.pipeline_cli
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.pipeline_context
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: StudySpec

.. automodule:: xgb.pipeline_evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.pipeline_reports
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xgb.pipeline_sweeps
   :members:
   :undoc-members:
   :show-inheritance:

KNN Baseline API
================

Feature Spaces
--------------

The ``knn`` command-line entry point exposes a ``--feature-space`` switch to
toggle between sparse TF-IDF queries (default), dense Word2Vec embeddings, and
Sentence-Transformer embeddings. Word2Vec support depends on ``gensim`` being
available; Sentence-Transformer support depends on the
``sentence-transformers`` package (and a compatible PyTorch backend). Install
the extras with ``pip install gensim sentence-transformers`` or include the
project’s ``dev`` extra.

Example invocations:

.. code-block:: bash

   python -m knn.cli --fit-index --feature-space tfidf --out-dir models/knn/run-tfidf

   python -m knn.cli --fit-index --feature-space word2vec \
       --word2vec-size 256 --out-dir models/knn/run-w2v

   python -m knn.cli --fit-index --feature-space sentence_transformer \
       --sentence-transformer-model sentence-transformers/all-MiniLM-L6-v2 \
       --out-dir models/knn/run-st

Diagnostics (elbow plots, accuracy-by-k curves, and per-k predictions) are
saved under the configured `--out-dir`; use `--train-curve-max` to control how
many training examples contribute to the curve metrics.

Elbow & Curve Metrics
---------------------

Each evaluation issue produces:

- ``reports/knn/next_video/<feature-space>/elbow_<issue>.png`` – accuracy vs. ``k`` with the
  elbow-selected point highlighted.
- ``models/knn/next_video/<feature-space>/<study>/<issue>/knn_curves_<issue>.json`` – serialised
  curves for the evaluation split (and optionally the training split) containing accuracy-by-``k``,
  eligible/correct counts, AUC (both absolute and normalised), and the selected ``k``.
- ``models/knn/next_video/<feature-space>/<study>/<issue>/`` – per-``k`` predictions and aggregate
  metrics for downstream analysis.

Set ``--train-curve-max`` when invoking the CLI to limit how many training
examples feed into the training curve diagnostics.

Package Overview
----------------

.. automodule:: knn
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.features
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: CandidateMetadata, Word2VecConfig

.. automodule:: knn.index
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.opinion
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. automodule:: knn.cli
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: DEFAULT_KNN_TEXT_FIELDS, add_sentence_transformer_normalize_flags, build_parser, main

Core (Canonical Paths)
----------------------

.. automodule:: knn.core.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.index
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.opinion
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: OpinionEmbeddingConfigs, OpinionEvaluationContext, OpinionExample, OpinionIndex, OpinionSpec, DEFAULT_SPECS

.. automodule:: knn.core.opinion_models
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: OpinionSpec

.. automodule:: knn.core.opinion_data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.opinion_index
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.opinion_predictions
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.opinion_plots
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.opinion_outputs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.core.utils
   :members:
   :undoc-members:
   :show-inheritance:

Core Utilities
--------------

.. automodule:: knn.utils
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Modules
----------------

.. automodule:: knn.pipeline.context
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:
   :exclude-members: StudySpec, ReportSelections, ReportOutcomes, ReportMetrics, ReportPresentation

.. automodule:: knn.pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: PipelineContext, ReportBundle

.. automodule:: knn.pipeline.__main__
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.cli
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.evaluate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.io
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.opinion_sweeps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.sweeps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.utils
   :members:
   :undoc-members:
   :show-inheritance:

Context Internals
-----------------

.. automodule:: knn.pipeline.context_pipeline
   :members:
   :undoc-members:
   :show-inheritance:
   :private-members:

.. automodule:: knn.pipeline.context_reports
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.context_sweeps
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.context_config
   :members:
   :undoc-members:
   :show-inheritance:

Report Builders
---------------

.. automodule:: knn.pipeline.reports
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.catalog
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.hyperparameter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: NextVideoReportInputs

.. automodule:: knn.pipeline.reports.opinion
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.shared
   :members:
   :undoc-members:
   :show-inheritance:

Next-Video Report Internals
---------------------------

.. automodule:: knn.pipeline.reports.next_video.inputs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.sections
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.curves
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.comparison
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.loso
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.csv_exports
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.helpers
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline.reports.next_video.report
   :members:
   :undoc-members:
   :show-inheritance:

Legacy Report Shims
-------------------

.. automodule:: knn.pipeline_reports
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline_reports.catalog
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline_reports.next_video
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline_reports.shared
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline_reports.features
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline_reports.hyperparameter
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.pipeline_reports.opinion
   :members:
   :undoc-members:
   :show-inheritance:

Scripts
-------

.. automodule:: knn.scripts
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.scripts.baseline
   :members:
   :undoc-members:
   :show-inheritance:

CLI Internals
-------------

.. automodule:: knn.cli.__main__
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.cli.main
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.cli.utils
   :members:
   :undoc-members:
   :show-inheritance:

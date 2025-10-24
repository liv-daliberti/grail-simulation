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

.. automodule:: knn.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: knn.features
   :members:
   :undoc-members:
   :show-inheritance:

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

.. automodule:: knn.cli
   :members:
   :undoc-members:
   :show-inheritance:

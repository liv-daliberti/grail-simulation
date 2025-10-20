KNN Baseline API
================

Feature Spaces
--------------

The ``knn`` command-line entry point now exposes a ``--feature-space`` switch to
toggle between sparse TF-IDF queries (default) and dense Word2Vec embeddings.
Word2Vec support depends on ``gensim`` being available; install it via
``pip install gensim`` or include the project’s ``dev`` extra.

Example invocations:

.. code-block:: bash

   python -m knn.cli --fit-index --feature-space tfidf --out-dir models/knn/run-tfidf

   python -m knn.cli --fit-index --feature-space word2vec \
       --word2vec-size 256 --out-dir models/knn/run-w2v

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

.. automodule:: knn.cli
   :members:
   :undoc-members:
   :show-inheritance:

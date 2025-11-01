GRPO Pipeline API
=================

The GRPO evaluation pipeline is composed of several small modules that
coordinate how prompts, paths, and stage selection are resolved before being
handed off to the stage runners. The entries below are indexed for quick
reference when integrating the pipeline in experiments or scripting end-to-end
runs.

Core Orchestration
------------------

.. automodule:: grpo.pipeline
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: grpo.pipeline_cli
   :members:
   :undoc-members:
   :show-inheritance:

Setup & Context
---------------

.. automodule:: grpo.pipeline_setup
   :members:
   :undoc-members:
   :show-inheritance:


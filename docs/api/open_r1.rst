Open-R1 Training API
====================

The :mod:`open_r1` package hosts reinforcement learning pipelines for
GRAIL's Open R1 experiments along with a suite of utilities for data
loading, rollout orchestration, and evaluation.  Use the modules below
to explore the available configuration surfaces and helper tooling.

Reward Output Format
--------------------

The GRPO prompts built by :mod:`open_r1` now expect **two** committed
answers from the policy.  Every completion must follow this layout::

   <think>…</think><answer>2</answer><opinion>decrease</opinion>

* ``<answer>…</answer>`` continues to identify the next-video choice by
  its 1-based index in the slate.
* ``<opinion>…</opinion>`` captures the viewer's predicted opinion
  direction and must be one of ``increase``, ``decrease``, or
  ``no_change``.

The training dataset exposes this supervision via the
``opinion_direction`` column, and :func:`open_r1.rewards.pure_accuracy_reward`
now awards partial credit when a completion correctly predicts one task
but not the other.  See the reward module API below for additional
details.

Core Modules
------------

.. automodule:: open_r1
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.configs
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.generate
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.grail
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.grpo
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.rewards
   :members:
   :undoc-members:
   :show-inheritance:

Utility Modules
---------------

.. automodule:: open_r1.utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.utils.callbacks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.utils.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.utils.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.utils.hub
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.utils.model_utils
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: open_r1.utils.replay_buffer
   :members:
   :undoc-members:
   :show-inheritance:

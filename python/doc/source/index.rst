.. CRAAM documentation master file, created by
   sphinx-quickstart on Mon Jul 25 17:36:09 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

CRAAM
=====

.. automodule:: craam.crobust

.. toctree::
   :maxdepth: 2

.. autosummary::
    :toctree: DIRNAME

    craam.MDP
    craam.RMDP
    craam.DiscreteSamples
    craam.SampledMDP
    craam.SimulatorMDP
    craam.MDPIR

Solving Markov Decision Processes
---------------------------------

Classes for modeling and solving regular Markov decision processes

.. autosummary::
    :toctree: DIRNAME

    craam.MDP

Class details:

.. autoclass:: craam.MDP
    :members:

Solving Robust Markov Decision Processes
----------------------------------------

Classes for modeling and solving robust Markov decision processes

.. autosummary::
    :toctree: DIRNAME

    craam.RMDP

Class details:

.. autoclass:: craam.RMDP
    :members:


Domain Samples and Constructing MDPs
------------------------------------

Classes for representing domain samples and constructing MDPs based on them.

.. autosummary::
    :toctree: DIRNAME

    craam.DiscreteSamples
    craam.SampledMDP

Class details:

.. autoclass:: craam.DiscreteSamples
    :members:

.. autoclass:: craam.SampledMDP
    :members:


Simulating Markov Decision Processes
------------------------------------

Classes for modeling and solving regular Markov decision processes

.. autosummary::
    :toctree: DIRNAME

    craam.SimulatorMDP

Class details:

.. autoclass:: craam.SimulatorMDP
    :members:

Solving Interpretable MDPs
--------------------------

Classes for solving interpretable MDPs

See:

- Petrik, M., & Luss, R. (2016). Interpretable Policies for Dynamic Product Recommendations. In Uncertainty in Artificial Intelligence (UAI).

.. autosummary::
    :toctree: DIRNAME

    craam.MDPIR

Class details:

.. autoclass:: craam.MDPIR
    :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


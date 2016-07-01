.. image:: https://travis-ci.org/marekpetrik/CRAAM.svg
    :target: https://travis-ci.org/marekpetrik/CRAAM

Robust And Approximate Markov decision processes
===============================================

.. role:: cpp(code)
    :language: c++

Craam is a C++ library for solving *plain*, *robust*, or *optimistic* Markov decision processes. The library also provides basic tools that enable simulation and construction of MDPs from samples. There is also support for state aggregation and abstraction solution methods. 

The library supports standard finite or infinite horizon discounted MDPs [Puterman2005]. Some basic stochazstic shortest path methods are also supported. The library assumes *maximization* over actions. The states and actions must be finite.

The robust model extends the regular MDPs [Iyengar2005]. The library allows to model uncertainty in *both* the transitions and rewards, unlike some published papers on this topic. This is modeled by adding an outcome to each action. The outcome is assumed to be minimized by nature, similar to [Filar1997].

In summary, the MDP problem being solved is:

.. math::

    v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s\in\mathcal{S}} ( r(s,a,o,s') + \gamma P(s,a,o,s') v(s') ) ~.

Here, :math:`\mathcal{S}` are the states, :math:`\mathcal{A}` are the actions, :math:`\mathcal{O}` are the outcomes. 

Available algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution.

Installation
------------

The library has minimal dependencies and should compile on all major operating systems.

Minimal Requirements
~~~~~~~~~~~~~~~~~~~~

- `CMake <http://cmake.org/>`__ 3.1.0
- C++14 compatible compiler 
    - Tested with Linux GCC 4.9.2,5.2.0,6.1.0; does not work with GCC 4.7, 4.8. 
    - Tested with Linux Clang 3.6.2 (and maybe 3.2+).
- `Boost <http://boost.org>`__ to enable unit tests and for some simple numerical algebra

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

- `OpenMP <http://openmp.org>`__ to enable parallel computation 
- `Doxygen <http://doxygen.org>`__  1.8.0+ to generate documentation

Build Instructions
~~~~~~~~~~~~~~~~~~

Build all default supported targets:

.. code:: bash

    $ cmake .
    $ cmake --build .

Run unit tests
~~~~~~~~~~~~~~

Note that Boost must be present in order to build the tests in the first place.

.. code:: bash

    $ cmake .
    $ cmake --build . --target testit

Build a benchmark executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a benchmark problem, download and decompress one of the following test files:

* Small problem with 100 states: https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip
* Medium problem with 2000 states (7zip): https://www.dropbox.com/s/k0znc23xf9mpe5i/ms.7z

These two benchmark problems were generated randomly.

The small benchmark example, for example, can be executed as follows:

.. code:: bash
    
    $ cmake --build . --target benchmark
    $ mkdir data
    $ cd data
    $ wget https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip
    $ unzip ss.zip
    $ cd ..
    $ bin/benchmark data/smallsize_test.csv
    
Development
~~~~~~~~~~~

CMake can generate project files for a variety of IDE's. For more see:

.. code:: bash

    $ cmake --help

Getting Started
---------------

The main interface to the library is through the class ``RMDP``. The class supports simple construction of an MDP and several methods for solving them. 

States, actions, and outcomes are identified using 0-based contiguous indexes. The actions are indexed independently for each states and the outcomes are indexed independently for each state and action pair. 

Transitions are added through functions :cpp:`RMDP::add_transition` and :cpp:`RMDP::add_transition_d`. The object is automatically resized according to the new transitions added. The actual algorithms are solved using:

======================  ====================================
Method                  Algorithm
======================  ====================================
:cpp:`RMDP::vi_gs_*`      Gauss-Seidel value iteration; runs in a single thread. Computes the worst-case outcome for each action.
:cpp:`RMDP::vi_jac_*`     Jacobi value iteration; parallelized with OpenMP. Computes the worst-case outcome for each action.
:cpp:`RMDP::mpi_jac_*`    Jacobi modified policy iteration; parallelized with OpenMP. Computes the worst-case outcome for each action. Generally, modified policy iteration is vastly more efficient than value iteration.
:cpp:`GRMDP::vi_jac_fix`     Jacobi value iteration for policy evaluation; parallelized with OpenMP. Computes the worst-case outcome for each action.

======================  ====================================


The following is a simple example of formulating and solving a small MDP. 

.. code:: c++

    #include "RMDP.hpp"
    #include "modeltools.hpp"

    #include <iostream>
    #include <vector>

    using namespace craam;

    int main(){
        MDP mdp(3);

        // transitions for action 0
        add_transition(mdp,0,0,0,1,0);
        add_transition(mdp,1,0,0,1,1);
        add_transition(mdp,2,0,1,1,1);

        // transitions for action 1
        add_transition(mdp,0,1,1,1,0);
        add_transition(mdp,1,1,2,1,0);
        add_transition(mdp,2,1,2,1,1.1);

        // solve using Jacobi value iteration
        auto&& re = mdp.mpi_jac(Uncertainty::Average,0.9);

        for(auto v : re.valuefunction){
            cout << v << " ";
        }

        return 0;
    }

To compile the file, run:

.. code:: bash
    
     $ g++ -std=c++11 -I<path_to_RAAM.h> -L . -lcraam simple.cpp


Documentation
-------------

The documentation can be generated using `doxygen <http://www.stack.nl/~dimitri/doxygen/>`_; the configuration file and the documentation are in the ``doc`` directory.

General Assumptions
~~~~~~~~~~~~~~~~~~~

* Transition probabilities must be non-negative but do not need to add up to a specific value
* Transitions with 0 probabilities may be omitted, except there must be at least one target state in each transition
* State with no actions: A terminal state with value 0
* Action with no outcomes: Terminates with an error
* Outcome with no target states: Terminates with an error

Common Use Cases
----------------

1. Formulate an uncertain MDP
2. Compute a solution to an uncertain MDP
3. Compute value of a fixed policy
4. Compute an occupancy frequency
5. Simulate transitions of an MDP
6. Construct MDP from samples
7. Simulate a general domain

References
----------

.. [Filar1997] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.

.. [Puterman2005] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.

.. [Iyengar2005] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29.

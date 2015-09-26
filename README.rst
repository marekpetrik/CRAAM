Robust And Approximate Markov decision processes
===========================================

.. role:: cpp(code)
    :language: c++

A simple and easy to use C++ library to solve Markov decision processes and *robust* Markov decision processes. 

The library supports standard finite or infinite horizon discounted MDPs [Puterman2005]_. The library assumes *maximization* over actions. The states and actions must be finite.

The robust model extends the regular MDPs [Iyengar2005]_. The library allows to model uncertainty in *both* the transition and rewards, unlike some published papers on this topic. This is modeled by adding an outcome to each action. The outcome is assumed to be minimized by nature, similar to [Filar1997]_.

In summary, the MDP problem being solved is:

.. math::

    v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s\in\mathcal{S}} ( r(s,a,o,s') + \gamma P(s,a,o,s') v(s') ) ~.

Here, :math:`\mathcal{S}` are the states, :math:`\mathcal{A}` are the actions, :math:`\mathcal{O}` are the outcomes. 

The included algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution (see methods of :cpp:`RMDP` that end with :cpp:`_l1`).

Installation
------------

The library has minimal dependencies and should compile on all major operating systems.

Minimal Requirements
~~~~~~~~~~~~~~~~~~~~

* `CMake <http://cmake.org/>` 3.1.0
* C++11 compatible compiler

The main dependence is a compiler that supports the c++14 standard (e.g., gcc 4.9 or later, or clang 3.4 or later). Everything except Simulation.hpp (only required to run simulations) will also compile using C++11 standard (gcc 4.7 or later). A compiler with OpenMP support is needed to run the computation on multiple cores. The tests included with the library need `boost libraries <http://boost.org>`_ version at least 1.21 to run. 

The code has been tested on:

* Linux
* GCC 5.2.0
* Boost 1.58.0
* GNU Make 4.1

**Note**: The project will also compile with GCC 4.6, but requires that ``-std=c++11`` is replaced by ``-std=c++0x`` and the constructor :cpp:`RMDP::RMDP()` needs to be implemented explicitely (without calling :cpp:`RMDP::RMDP(long)`). 

There is a makefile included in the project. 
   
Build a static library
~~~~~~~~~~~~~~~~~~~~~~

To build a shared library:

.. code:: bash

    $ make release

Build tests
~~~~~~~~~~~

To run the tests:

.. code:: bash

    $ make test

Build a benchmark executable
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To run a benchmark problem, download and decompress one of the following test files:

* Small problem with 100 states: https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip
* Medium problem with 2000 states (7zip): https://www.dropbox.com/s/k0znc23xf9mpe5i/ms.7z

These two benchmark problems were generated randomly.

The small benchmark, for example, can be executed as follows:

.. code:: bash
    
    $ wget https://www.dropbox.com/s/b9x8sz7q5ow1vm4/ss.zip
    $ unzip ss.zip
    $ make benchmark
    $ bin/Benchmark/raam smallsize_test.csv
    
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
:cpp:`RMDP::vi_gs_l1_*`   The same as ``vi_gs`` except the worst case is bounded with respect to an :math:`L_1` norm.
:cpp:`RMDP::vi_jac_l1_*`  The same as ``vi_jac`` except the worst case is bounded with respect to an :math:`L_1` norm.
:cpp:`RMDP::mpi_jac_*`    Jacobi modified policy iteration; parallelized with OpenMP. Computes the worst-case outcome for each action. Generally, modified policy iteration is vastly more efficient than value iteration.
======================  ====================================

The star in the above can be one of {:cpp:`rob`, :cpp:`opt`, :cpp:`ave`} which represents the actions of nature. The values represent respective the worst case (robust), the best case (optimistic), and average.

The following is a simple example of formulating and solving a small MDP. 

.. code:: c++

    #include <iostream>
    #include <vector>
    #include "RMDP.h"
    
    use namespace craam;
    
    int main(){
        RMDP rmdp(3);

        // transitions for action 0
        rmdp.add_transition_d(0,0,0,1,0);
        rmdp.add_transition_d(1,0,0,1,1);
        rmdp.add_transition_d(2,0,1,1,1);

        // transitions for action 1
        rmdp.add_transition_d(0,1,1,1,0);
        rmdp.add_transition_d(1,1,2,1,0);
        rmdp.add_transition_d(2,1,2,1,1.1);
    
        // prec_t is the numeric precision type used throughout the library (double)
        vector<prec_t> initial{0,0,0};
    
        // solve using Jacobi value iteration
        auto&& re = rmdp.vi_jac_rob(initial,0.9,20,0);
    
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


References
----------

.. [Filar1997] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.

.. [Puterman2005] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.

.. [Iyengar2005] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29.

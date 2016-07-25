#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <assert.h>

#include "definitions.hpp"

using namespace std;

namespace craam {

/** \mainpage

Introduction
------------

Craam is a C++ library for solving *plain*, *robust*, or *optimistic* Markov decision processes. The library also provides basic tools that enable simulation and construction of MDPs from samples. There is also support for state aggregation and abstraction solution methods. 

The library supports standard finite or infinite horizon discounted MDPs [Puterman2005]. Some basic stochazstic shortest path methods are also supported. The library assumes *maximization* over actions. The states and actions must be finite.

The robust model extends the regular MDPs [Iyengar2005]. The library allows to model uncertainty in *both* the transitions and rewards, unlike some published papers on this topic. This is modeled by adding an outcome to each action. The outcome is assumed to be minimized by nature, similar to [Filar1997].

In summary, the MDP problem being solved is:

\f[v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s\in\mathcal{S}} ( r(s,a,o,s') + \gamma P(s,a,o,s') v(s') ) ~.\f]

Here, \f$\mathcal{S}\f$ are the states, \f$\mathcal{A}\f$ are the actions, \f$\mathcal{O}\f$ are the outcomes.

Available algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution.

Installation and Build Instruction
----------------------------------

See the README.rst

Getting Started
---------------

The main interface to the library is through the templated class GRMDP. The templated version of this class enable different definitions of the uncertainty set. The avialable specializations are:

- craam::MDP : plain MDP with no definition of uncertainty
- craam::RMDP_D : a robust/uncertain with discrete outcomes with the best/worst one chosen
- craam::RMDP_L1 : a robust/uncertain with discrete outcomes with L1 constraints on the uncertainty


States, actions, and outcomes are identified using 0-based contiguous indexes. The actions are indexed independently for each states and the outcomes are indexed independently for each state and action pair.

Transitions are added through function add_transition. New states, actions, or outcomes are automatically added based on the new transition. The actual algorithms are solved using:

| Method                  |  Algorithm     |
| ----------------------- | ----------------
| GRMDP::vi_gs            | Gauss-Seidel value iteration; runs in a single thread. Computes the worst-case outcome for each action.
| GRMDP::vi_jac           | Jacobi value iteration; parallelized with OpenMP. Computes the worst-case outcome for each action.
| GRMDP::mpi_jac          | Jacobi modified policy iteration; parallelized with OpenMP. Computes the worst-case outcome for each action. Generally, modified policy iteration is vastly more efficient than value iteration.
| GRMDP::vi_jac_fix       | Jacobi value iteration for policy evaluation; parallelized with OpenMP. Computes the worst-case outcome for each action.


For uncertain MDPs, each method supports average, robust, and optimistic computation modes.

The following is a simple example of formulating and solving a small MDP.

\code

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

\endcode

To compile the file, run:

\code{.sh}
     $ g++ -std=c++11 -I<path_to_RAAM.h> -L . -lcraam simple.cpp
\endcode

Common Use Cases
----------------

1. Formulate an uncertain MDP
2. Compute a solution to an uncertain MDP
3. Compute value of a fixed policy
4. Compute an occupancy frequency
5. Simulate transitions of an MDP
6. Construct MDP from samples
7. Simulate a general domain


General Assumptions
-------------------

- Transition probabilities must be non-negative but do not need to add up to one
- Transitions with 0 probabilities may be omitted, except there must be at least one target state in each transition
- **State with no actions**: A terminal state with value 0
- **Action with no outcomes**: Terminates with an error
- **Outcome with no target states**: Terminates with an error


References
----------

[Filar1997] Filar, J., & Vrieze, K. (1997). Competitive Markov decision processes. Springer.

[Puterman2005] Puterman, M. L. (2005). Markov decision processes: Discrete stochastic dynamic programming. Handbooks in operations research and management …. John Wiley & Sons, Inc.

[Iyengar2005] Iyengar, G. N. G. (2005). Robust dynamic programming. Mathematics of Operations Research, 30(2), 1–29.

[Petrik2014] Petrik, M., Subramanian S. (2014). RAAM : The benefits of robustness in approximating aggregated MDPs in reinforcement learning. In Neural Information Processing Systems (NIPS).

[Petrik2016] Petrik, M., & Luss, R. (2016). Interpretable Policies for Dynamic Product Recommendations. In Uncertainty in Artificial Intelligence (UAI).
*/


template <typename T> vector<size_t> sort_indexes(vector<T> const& v) {
    /** \brief Sort indices by values in ascending order
     *
     * \param v List of values
     * \return Sorted indices
     */

    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

template vector<size_t> sort_indexes<long>(vector<long> const&);

template <typename T> vector<size_t> sort_indexes_desc(vector<T> const& v)
{
    /** \brief Sort indices by values in descending order
     *
     * \param v List of values
     * \return Sorted indices
     */

    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

pair<numvec,prec_t> worstcase_l1(numvec const& z, numvec const& q, prec_t t){
    /**
    Computes the solution of:
    min_p   p^T * z
    s.t.    ||p - q|| <= t
            1^T p = 1
            p >= 0

    Notes
    -----
    This implementation works in O(n log n) time because of the sort. Using
    quickselect to choose the right quantile would work in O(n) time.

    This function does not check whether the probability distribution sums to 1.
    **/

    assert(*min_element(q.begin(), q.end()) >= 0 && *max_element(q.begin(), q.end()) <= 1);
    assert(z.size() > 0);
    assert(t >= 0.0 && t <= 2.0);
    assert(z.size() == q.size());

    size_t sz = z.size();

    vector<size_t> smallest = sort_indexes<prec_t>(z);
    numvec o(q);

    auto k = smallest[0];
    auto epsilon = min(t/2, 1-q[k]);

    o[k] += epsilon;

    auto i = sz - 1;
    while(epsilon > 0){
        k = smallest[i];
        auto diff = min( epsilon, o[k] );
        o[k] -= diff;
        epsilon -= diff;
        i -= 1;
    }

    auto r = inner_product(o.begin(),o.end(),z.begin(), (prec_t) 0.0);

    return make_pair(move(o),r);
}


}

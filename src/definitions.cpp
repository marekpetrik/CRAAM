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

A simple and easy to use C++ library to solve Markov decision processes and *robust* Markov decision processes.

The library supports standard finite or infinite horizon discounted MDPs [Puterman2005]. The library assumes *maximization* over actions. The states and actions must be finite.

The robust model extends the regular MDPs [Iyengar2005]. The library allows to model uncertainty in *both* the transitions and rewards, unlike some published papers on this topic. This is modeled by adding an outcome to each action. The outcome is assumed to be minimized by nature, similar to [Filar1997].

In summary, the MDP problem being solved is:

\f[v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s\in\mathcal{S}} ( r(s,a,o,s') + \gamma P(s,a,o,s') v(s') ) ~.\f]

Here, \f$\mathcal{S}\f$ are the states, \f$\mathcal{A}\f$ are the actions, \f$\mathcal{O}\f$ are the outcomes.

The included algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution (see methods of RMDP that end with _l1).

Installation and Build Instruction
----------------------------------

See the github/bitbucket README file

Getting Started
---------------

The main interface to the library is through the class RMDP. The class supports simple construction of an MDP and several methods for solving them.

States, actions, and outcomes are identified using 0-based contiguous indexes. The actions are indexed independently for each states and the outcomes are indexed independently for each state and action pair.

Transitions are added through functions RMDP::add_transition and RMDP::add_transition_d. The object is automatically resized according to the new transitions added. The actual algorithms are solved using:

| Method                  |  Algorithm     |
| ----------------------- | ----------------
| RMDP::vi_gs_*           |  Gauss-Seidel value iteration; runs in a single thread. Computes the worst-case outcome for each action. |
| RMDP::vi_jac_*          |  Jacobi value iteration; parallelized with OpenMP. Computes the worst-case outcome for each action. |
| RMDP::vi_gs_l1_*        |  The same as vi_gs except the worst case is bounded with respect to an L1 norm. |
| RMDP::vi_jac_l1_*       |    The same as vi_jac except the worst case is bounded with respect to an L1 norm. |

The star in the above can be one of {rob, opt, ave} which represents the actions of nature. The values represent respective the worst case (robust), the best case (optimistic), and average.

The following is a simple example of formulating and solving a small MDP.

\code
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

\endcode

To compile the file, run:

\code{.sh}
     $ g++ -std=c++11 -I<path_to_RAAM.h> -L . -lcraam simple.cpp
\endcode

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

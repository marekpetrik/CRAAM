// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


#pragma once

#include "State.hpp"

#include <rm/range.hpp>
#include <vector>
#include <istream>
#include <fstream>
#include <memory>
#include <tuple>
#include <cassert>
#include <limits>
#include <algorithm>
#include <string>
#include <sstream>
#include <utility>
#include <iostream>


/** \mainpage

Introduction
------------

Craam is a **header-only** C++ library for solving  Markov decision processes with *regular*, *robust*, or *optimistic* objectives. The optimistic objective is the opposite of robust, in which nature chooses the best possible realization of the uncertain values. The library also provides tools for *basic simulation*, for constructing MDPs from *sample*s, and *value function approximation*. Objective functions supported are infinite horizon discounted MDPs, finite horizon MDPs, and stochastic shortest path [Puterman2005]. Some basic stochastic shortest path methods are also supported. The library assumes *maximization* over actions. The number of states and actions must be finite.
 
The library is build around two main data structures: MDP and RMDP. **MDP** is the standard model that consists of states $\mathcal{S}$ and actions $\mathcal{A}$. The robust solution for an MDP would satisfy, for example, the following Bellman optimality equation:
\f[ v(s) = \max_{a \in \mathcal{A}} \min_{p \in \Delta} \left\{ \sum_{s'\in\mathcal{S}} p(s') ( r(s,a,s') + \gamma \,  \, v(s') ) ~:~ \|p - P(s,a,\cdot) \| \le \psi, \; p \ll P(s,a,\cdot) \right\}~. \f]
Note that $p$ is constrained to be **absolutely continuous** with respect to $P(s,a,\cdot)$. This is a hard requirement for all choices of ambiguity (or uncertainty). 
 
The **RMPD** model adds a set of *outcomes* that model possible actions that can be taken by nature. In that case, the robust solution may for example satisfy the following Bellman optimality equation:
\f[ v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s'\in\mathcal{S}} P(s,a,o,s')  ( r(s,a,o,s') + \gamma \, v(s') ) ~. \f]
Using outcomes makes it more convenient to capture correlations between the ambiguity in rewards and the uncertainty in transition probabilities. It also make it much easier to represent uncertainties that lie in small-dimensional vector spaces. The equation above uses the worst outcome, but in general distributions over outcomes are supported.
 
The available algorithms are *value iteration* and *modified policy iteration*. The library support both the plain worst-case outcome method and a worst case with respect to a base distribution.
 
Installation and Build Instruction
----------------------------------

See the README.rst

Getting Started
---------------

See the [online documentation](http://cs.unh.edu/~mpetrik/code/craam) or generate it locally as described above. 
 
Unit tests provide some examples of how to use the library. For simple end-to-end examples, see `tests/benchmark.cpp` and `test/dev.cpp`. Targets `BENCH` and `DEV` build them respectively.
 
The main models supported are:
 
- `craam::MDP` : plain MDP with no specific definition of ambiguity (can be used to compute robust solutions anyway)
- `craam::RMDP` : an augmented model that adds nature's actions (so-called outcomes) to the model for convenience
- `craam::impl::MDPIR` : an MDP with implementability constraints. See [Petrik2016].
 
The regular value-function based methods are in the header `algorithms/values.hpp` and the robust versions are in in the header `algorithms/robust_values.hpp`. There are 4 main value-function based methods:
 
| Method                  |  Algorithm                           |
| ----------------------- | ------------------------------------ |
| `solve_vi`                | Gauss-Seidel value iteration; runs in a single thread. 
| `solve_mpi`               | Jacobi modified policy iteration; parallelized with OpenMP. Generally, modified policy iteration is vastly more efficient than value iteration.
| `rsolve_vi`              | Like the value iteration above, but also supports robust, risk-averse, or optimistic objectives.
| `rsolve_mpi`              | Like the modified policy iteration above, but it also supports robust, risk-averse, optimistic objective.  
 
These methods can be applied to eithen an MDP or an RMDP.
 
The header `algorithms/occupancies.hpp` provides tools for converting the MDP to a transition matrix and computing the occupancy frequencies.
 
There are tools for building simulators and sampling from simulations in the header `Simulation.hpp` and methods for handling samples in `Samples.hpp`.

The following is a simple example of formulating and solving a small MDP.

\code

    #include "craam/RMDP.hpp"
    #include "craam/modeltools.hpp"
    #include "craam/algorithms/values.hpp"

    #include <iostream>
    #include <vector>

    using namespace craam;
    using namespace std;

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
        auto&& re = algorithms::solve_mpi(mdp,0.9);

        for(auto v : re.valuefunction){
            cout << v << " ";
        }

        return 0;
    }

\endcode

To compile the file, run:

\code{.sh}
     $ g++ -fopenmp -std=c++14 -I<path_to_top_craam_folder> simple.cpp
\endcode


Supported Use Cases
--------------------

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


/// Main namespace which includes modeling a solving functionality
namespace craam {

using namespace std;
using namespace util::lang;

// **************************************************************************************
//  Generic MDP Class
// **************************************************************************************

/**
A general robust Markov decision process. Contains methods for constructing and solving RMDPs.

Some general assumptions (may depend on the state and action classes):
    - Transition probabilities must be non-negative but do not need to add
        up to a specific value
    - Transitions with 0 probabilities may be omitted, except there must
        be at least one target state in each transition
    - State with no actions: A terminal state with value 0
    - Action with no outcomes: Terminates with an error for uncertain models, but
                               assumes 0 return for regular models.
    - Outcome with no target states: Terminates with an error
    - Invalid actions are ignored
    - Behavior for a state with all invalid actions is not defined

\tparam SType Type of state, determines s-rectangularity or s,a-rectangularity and
        also the type of the outcome and action constraints
 */
template<class SType>
class GRMDP{
protected:
    /** Internal list of states */
    vector<SType> states;

public:

    /// Type of the state
    using state_type = SType;

    /** Decision-maker's policy: Which action to take in which state.  */
    typedef indvec policy_det;
    /** Nature's policy: Which outcome to take in which state.  */
    typedef vector<numvec> policy_rand;

    /**
    Constructs the RMDP with a pre-allocated number of states. All
    states are initially terminal.
    \param state_count The initial number of states, which dynamically
                        increases as more transitions are added. All initial
                        states are terminal.
    */
    GRMDP(long state_count) : states(state_count){};

    /** Constructs an empty RMDP. */
    GRMDP() : states() {};

    /**
    Assures that the MDP state exists and if it does not, then it is created.
    States with intermediate ids are also created
    \return The new state
    */
    SType& create_state(long stateid){
        assert(stateid >= 0);
        if(stateid >= (long) states.size())
            states.resize(stateid + 1);
        return states[stateid];
    }

    /**
    Creates a new state at the end of the states
    \return The new state
    */
    SType& create_state(){ return create_state(states.size());};

    /** Number of states */
    size_t state_count() const {return states.size();};

    /** Number of states */
    size_t size() const {return state_count();};

    /** Retrieves an existing state */
    const SType& get_state(long stateid) const {
        assert(stateid >= 0 && size_t(stateid) < state_count());
        return states[stateid];};

    /** Retrieves an existing state */
    const SType& operator[](long stateid) const {return get_state(stateid);};

    /** Retrieves an existing state */
    SType& get_state(long stateid) {
        assert(stateid >= 0 && size_t(stateid) < state_count());
        return states[stateid];};

    /** Retrieves an existing state */
    SType& operator[](long stateid){return get_state(stateid);};

    /** \returns list of all states */
    const vector<SType>& get_states() const {return states;};

    /**
    Check if all transitions in the process sum to one.
    Note that if there are no actions, or no outcomes for a state,
    the RMDP still may be normalized.
    \return True if and only if all transitions are normalized.
     */
    bool is_normalized() const{
        for(auto const& s : states){
            for(auto const& a : s.get_actions()){
                for(auto const& t : a.get_outcomes()){
                    if(!t.is_normalized()) return false;
        } } }
        return true;
    }

    /** Normalize all transitions to sum to one for all states, actions, outcomes. */
    void normalize(){
        for(SType& s : states)
            s.normalize();
    }

    /**
    Checks if the policy and nature's policy are both correct.
    Action and outcome can be arbitrary for terminal states.

    \tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic 
                policy and a randomized policy of the nature
    \param policies The policy (indvec) or the pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically 
        a randomized policy
    \return If incorrect, the function returns the first state with an incorrect
            action and outcome. Otherwise the function return -1.
    */
    template<typename Policy>
    long is_policy_correct(const Policy& policies) const{
        for(auto si : indices(states) ){
            // ignore terminal states
            if(states[si].is_terminal())
                continue;

            // call function of the state
            if(!states[si].is_action_correct(policies))
                return si;
        }
        return -1;
    }

    // ----------------------------------------------
    // Reading and writing files
    // ----------------------------------------------

    // string representation
    /**
    Returns a brief string representation of the RMDP.
    This method is mostly suitable for analyzing small RMDPs.
    */
    string to_string() const{
        string result;

        for(size_t si : indices(states)){
            const auto& s = get_state(si);
            result += (std::to_string(si));
            result += (" : ");
            result += (std::to_string(s.action_count()));
            result += ("\n");
            for(size_t ai : indices(s)){
                result += ("    ");
                result += (std::to_string(ai));
                result += (" : ");
                const auto& a = s.get_action(ai);
                a.to_string(result);
                result += ("\n");
            }
        }
        return result;
    }

    /**
    Returns a json representation of the RMDP.
    This method is mostly suitable to analyzing small RMDPs.
    */
    string to_json() const{
        string result{"{\"states\" : ["};
        for(auto si : indices(states)){
            const auto& s = states[si];
            result += s.to_json(si);
            result += ",";
        }
        if(!states.empty()) result.pop_back(); // remove last comma
        result += "]}";
        return result;

    }

    /**
     * Datermines which states and actions are invalid (have no transitions)
     * @return List of (state, action) pairs. An empty vector when all states and
     *          actions are valid
     */
    vector<pair<long,long>> invalid_state_actions() const{
        vector<pair<long,long>> invalid(0);
        for(size_t s = 0; s < states.size(); s++){
            indvec invalid_a = states[s].invalid_actions();
            for(size_t ia = 0; ia < invalid_a.size(); ia++){
                invalid.push_back(make_pair(s,ia));
            }
        }
        return invalid;
    }

    /**
     * Removes invalid actions, and reindexes the remaining ones accordingly.
     * @returns List of original action ids for each state
     */
    vector<indvec> pack_actions(){
        vector<indvec> result; result.reserve(size());

        for(SType& state : states){
            result.push_back(state.pack_actions());
        }
        return result;
    }

};

// **********************************************************************
// *********************    TEMPLATE DECLARATIONS    ********************
// **********************************************************************

/**
Regular MDP with discrete actions and one outcome per action
*/
typedef GRMDP<RegularState> MDP;

/**
An uncertain MDP with outcomes and weights. See craam::L1RobustState.
*/
typedef GRMDP<WeightedRobustState> RMDP;

}

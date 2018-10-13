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

/**
Value-function based methods (value iteration and policy iteration) style algorithms. 
Provides abstractions that allow generalization to both robust and regular MDPs.
*/ 
#pragma once

#include "craam/RMDP.hpp"
#include <rm/range.hpp>
#include <functional>
#include <type_traits>
#include <chrono>

/// Main namespace for algorithms that operate on MDPs and RMDPs
namespace craam::algorithms{

using namespace std;
using namespace util::lang;


// *******************************************************
// RegularAction computation methods
// *******************************************************

/**
Computes the average value of the action.

@param action Action for which to compute the value
@param valuefunction State value function to use
@param discount Discount factor
@return Action value
*/
inline prec_t value_action(const RegularAction& action, const numvec& valuefunction,
        prec_t discount) {
    return action.get_outcome().value(valuefunction, discount);
}

/**
Computes a value of the action for a given distribution. This function can be used
to evaluate a robust solution which may modify the transition probabilities.

The new distribution may be non-zero only for states for which the original distribution is
not zero.

@param action Action for which to compute the value
@param valuefunction State value function to use
@param discount Discount factor
@param distribution New distribution. The length must match the number of
            states to which the original transition probabilities are strictly greater than 0.
            The order of states is the same as in the underlying transition.
@return Action value
*/
inline prec_t value_action(const RegularAction& action, const numvec& valuefunction,
        prec_t discount, numvec distribution) {
    return action.get_outcome().value(valuefunction, discount, distribution);
}


// *******************************************************
// WeightedOutcomeAction computation methods
// *******************************************************

/**
Computes the average outcome using the provided distribution.

@param action Action for which the value is computed
@param valuefunction Updated value function
@param discount Discount factor
@return Mean value of the action
 */
inline prec_t value_action(const WeightedOutcomeAction& action, numvec const& valuefunction,
            prec_t discount) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if(action.get_outcomes().empty())
        throw invalid_argument("WeightedOutcomeAction with no outcomes");

    prec_t averagevalue = 0.0;
    const numvec& distribution = action.get_distribution();
    for(size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += distribution[i] * action[i].value(valuefunction, discount);
    return averagevalue;
}

/**
Computes the action value for a fixed index outcome.

@param action Action for which the value is computed
@param valuefunction Updated value function
@param discount Discount factor
@param distribution Custom distribution that is selected by nature.
@return Value of the action
 */
inline prec_t value_action(const WeightedOutcomeAction& action, numvec const& valuefunction,
                    prec_t discount, const numvec& distribution) {

    assert(distribution.size() == action.get_outcomes().size());
    if(action.get_outcomes().empty()) throw invalid_argument("WeightedOutcomeAction with no outcomes");

    prec_t averagevalue = 0.0;
    // TODO: simd?
    for(size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += distribution[i] * action[i].value(valuefunction, discount);
    return averagevalue;
}


// *******************************************************
// State computation methods
// *******************************************************

/**
Finds the action with the maximal average return. The return is 0 with no actions. Such
state is assumed to be terminal.

@param state State to compute the value for
@param valuefunction Value function to use for the following states
@param discount Discount factor

@return (Index of best action, value), returns 0 if the state is terminal.
*/
template<class AType>
inline pair<long,prec_t> value_max_state(const SAState<AType>& state, const numvec& valuefunction,
                                     prec_t discount) {
    if(state.is_terminal())
        return make_pair(-1,0.0);
    // skip invalid state.get_actions()



    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;

    for(size_t i = 0; i < state.size(); i++){
        auto const& action = state[i];

        if(!state.is_valid(i))
            throw invalid_argument("cannot have an invalid state and action");

        auto value = value_action(action, valuefunction, discount);
        if(value >= maxvalue){
            maxvalue = value;
            result = i;
        }
    }

    // if the result has not been changed, that means that all actions are invalid
    if(result == -1)
        throw invalid_argument("all actions are invalid.");

    return make_pair(result, maxvalue);
}

/**
Computes the value of a fixed (and valid) action. Performs validity checks.

@param state State to compute the value for
@param valuefunction Value function to use for the following states
@param discount Discount factor

@return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t value_fix_state(const SAState<AType>& state, numvec const& valuefunction,
                              prec_t discount, long actionid) {
    // this is the terminal state, return 0
    if(state.is_terminal())
        return 0;
    if(actionid < 0 || actionid >= (long) state.get_actions().size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + 
                            to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume invalid state.get_actions()
    if(!state.is_valid(actionid)) throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount);
}

/**
Computes the value of a fixed action and fixed response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actionid Action prescribed by the policy
@param distribution New distribution over states with non-zero nominal probabilities

@return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t
value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              long actionid, numvec distribution) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return 0;

    assert(actionid >= 0 && actionid < long(state.size()));

    //if(actionid < 0 || actionid >= long(state.size())) throw range_error("invalid actionid: "
    //    + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];

    return value_action(action, valuefunction, discount, distribution);
}

/**
Computes the value of a fixed action and fixed response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actiondist Distribution over actions
@param distribution New distribution over states with non-zero nominal probabilities

@return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t
value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              numvec actiondist, numvec distribution) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return 0;

    assert(actiondist.size() == state.size());
    assert((1.0 - accumulate(actiondist.cbegin(), actiondist.cend(), 0.0) - 1.0) < 1e-5);

    prec_t result = 0.0;
    for(size_t actionid = 0; actionid < state.size(); actionid++){
        const auto& action = state[actionid];
        // cannot assume that the action is valid
        if(!state.is_valid(actionid)) throw invalid_argument("Cannot take an invalid action");

        result += actiondist[actionid] * value_action(action, valuefunction, discount, distribution);
    }
    return result;
}

/**
Computes the value of a fixed action and fixed response of nature.

@param state State to compute the value for
@param valuefunction Value function to use in computing value of states.
@param discount Discount factor
@param actiondist Distribution over actions
@param distributions New distribution over states with non-zero nominal probabilities and
                for actions that have a positive actiondist probability

@return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t
value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              numvec actiondist, vector<numvec> distributions) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return 0;

    assert(actiondist.size() == state.size());
    assert((1.0 - accumulate(actiondist.cbegin(), actiondist.cend(), 0.0) - 1.0) < 1e-5);
    assert(distributions.size() == actiondist.size());

    prec_t result = 0.0;
    for(size_t actionid = 0; actionid < state.size(); actionid++){
        const auto& action = state[actionid];
        // cannot assume that the action is valid
        if(!state.is_valid(actionid)) throw invalid_argument("Cannot take an invalid action");

        if(actiondist[actionid] <= EPSILON)
            continue;

        result += actiondist[actionid] * value_action(action, valuefunction, discount, distributions[actionid]);
    }
    return result;
}

// *******************************************************
// RMDP computation methods
// *******************************************************

/**
 * A set of values that represent a solution to a plain MDP.
 *
 * @tparam PolicyType Type of the policy used (int deterministic, numvec stochastic,
 *          but could also have multiple components (such as an action and transition probability) )
*/
template<class PolicyType>
struct Solution {
    /// Value function
    numvec valuefunction;
    /// Policy of the decision maker (and nature if applicable) for each state
    vector<PolicyType> policy;
    /// Bellman residual of the computation
    prec_t residual;
    /// Number of iterations taken
    long iterations;
    /// Time taken to solve the problem
    prec_t time;

    Solution(): valuefunction(0), policy(0), residual(-1),iterations(-1), time(nan("")) {}

    /// Empty solution for a problem with statecount states
    Solution(size_t statecount): valuefunction(statecount, 0.0), policy(statecount),
                                residual(-1), iterations(-1), time(nan("")) {}

    /// Empty solution for a problem with a given value function and policy
    Solution(numvec valuefunction, vector<PolicyType> policy, prec_t residual = -1, long iterations = -1,
             double time = nan("")) :
        valuefunction(move(valuefunction)), policy(move(policy)),
        residual(residual), iterations(iterations),
        time(time) {}

    /**
    Computes the total return of the solution given the initial
    distribution.
    @param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const{
        if(initial.max_index() >= (long) valuefunction.size()) throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};

/// A solution with a deterministic policy
using DeterministicSolution = Solution<long>;

// **************************************************************************
// Helper classes to handle computing of the best response
// **************************************************************************

/**
A Bellman update class for solving regular Markov decision processes. This class abstracts
away from particular model properties and the goal is to be able to plug it in into value or
policy iteration methods for MDPs.

Many of the methods are parametrized by the type of the state.

The class also allows to use an initial policy specification. See the constructor for the
definition.
*/
template<class SType = RegularState>
class PlainBellman{
public:
    /// Provides the type of policy for each state (int represents a deterministic policy)
    using policy_type = long;
    using state_type = SType;

    /// Constructs the update with no constraints on the initial policy
    PlainBellman() : initial_policy(0) {}

    /** A partial policy that can be used to fix some actions
     *  @param policy policy[s] = -1 means that the action should be optimized in the state
     *                policy of length 0 means that all actions will be optimized
     */
    PlainBellman(indvec policy) : initial_policy(move(policy)) {}

    /**
     *  Computes the Bellman update and returns the optimal action.
     *  @returns New value for the state and the policy
     */
    pair<prec_t,policy_type> policy_update(const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{

        // check whether this state should only be evaluated
        if(initial_policy.empty() || initial_policy[stateid] < 0){    // optimizing
            prec_t newvalue;
            policy_type action;

            tie(action, newvalue) = value_max_state(state, valuefunction, discount);
            return make_pair(newvalue,action);
        }else{// fixed-action, do not copy
            return {value_fix_state(state, valuefunction, discount, initial_policy[stateid]),initial_policy[stateid]};
        }
    }

    /**
     *  Computes value function update using the current policy
     * @returns New value for the state
     */
    prec_t compute_value(const policy_type& action, const SType& state, const numvec& valuefunction,
                            prec_t discount) const{

        return value_fix_state(state, valuefunction, discount, action);
    }
protected:
    /// Partial policy specification (action -1 is ignored and optimized)
    const indvec initial_policy;
};


// **************************************************************************
// Main solution methods
// **************************************************************************

/**
Gauss-Seidel variant of value iteration (not parallelized). See solve_vi for a simplified interface.

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the last state to the first one, the states should be ordered
in the temporal order.

@param mdp The mdp to solve
@param discount Discount factor.
@param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
@param response Using PolicyResponce allows to specify a partial policy. Only the actions that
                not provided by the partial policy are included in the optimization. 
                Using a class of a different types enables computing other objectives,
                such as robust or risk averse ones. 
@param iterations Maximal number of iterations to run
@param maxresidual Stop when the maximal residual falls below this value.

@tparam SType Type of the state used
@tparam ResponseType Class responsible for computing the Bellman updates. Should be
                     compatible with PlainBellman

@returns Solution that can be used to compute the total return, or the optimal policy.
 */
template<class SType, class ResponseType = PlainBellman<SType>>
inline Solution<typename ResponseType::policy_type>
vi_gs(const GRMDP<SType>& mdp, prec_t discount,
                        numvec valuefunction=numvec(0), const ResponseType& response = PlainBellman<SType>(),
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC)
                        {

    static_assert(std::is_same<SType,typename ResponseType::state_type>::value, "SType in vi_gs and the ResponseType passed to it must be the same");

    // time the computation
    auto start = chrono::steady_clock::now();

    using policy_type = typename ResponseType::policy_type;

    const auto& states = mdp.get_states();
    // just quit if there are no states
    if( mdp.state_count() == 0) return Solution<policy_type>(0);
    if(valuefunction.empty()){
        valuefunction.resize(mdp.state_count(), 0.0);
    }

    vector<policy_type> policy(mdp.state_count());

    // initialize values
    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;   // iterations defined outside to make them reportable

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            prec_t newvalue;
            tie(newvalue, policy[s]) = response.policy_update(states[s], long(s), valuefunction, discount);

            residual = max(residual, abs(valuefunction[s] - newvalue));
            valuefunction[s] = newvalue;
        }
    }

    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish-start;
    return Solution<policy_type>(move(valuefunction), move(policy),residual,i, duration.count());
}


/**
Modified policy iteration using Jacobi value iteration in the inner loop. See solve_mpi for 
a simplified interface.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
@param type Type of realization of the uncertainty
@param discount Discount factor
@param valuefunction Initial value function
@param response Using PolicyResponce allows to specify a partial policy. Only the actions that
                not provided by the partial policy are included in the optimization. 
                Using a class of a different types enables computing other objectives,
                such as robust or risk averse ones. 
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
@param iterations_vi Maximal number of inner loop value iterations
@param maxresidual_vi_rel Stop policy evaluation when the policy residual drops below
                            maxresidual_vi_rel * last_policy_residual

@param print_progress Whether to report on progress during the computation

@tparam SType Type of the state used
@tparam ResponseType Class responsible for computing the Bellman updates. Should be compatible
                      with PlainBellman

@return Computed (approximate) solution
 */
template<class SType, class ResponseType = PlainBellman<SType>>
inline Solution<typename ResponseType::policy_type>
mpi_jac(const GRMDP<SType>& mdp, prec_t discount,
                const numvec& valuefunction=numvec(0), const ResponseType& response = PlainBellman<SType>(),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi_rel=0.9,
                bool print_progress=false) {

    static_assert(std::is_same<SType,typename ResponseType::state_type>::value, "SType in mpi_jac and the ResponseType passed to it must be the same");

    // time the computation
    auto start = chrono::steady_clock::now();

    using policy_type = typename ResponseType::policy_type;

    const auto& states = mdp.get_states();

    // just quit if there are no states
    if( mdp.state_count() == 0) return Solution<policy_type>(0);

    // intialize the policy
    vector<policy_type> policy(mdp.state_count());

    numvec sourcevalue = valuefunction;         // value function to compute the update
    // resize if the the value function is empty and initialize to 0
    if(sourcevalue.empty()) sourcevalue.resize(mdp.state_count(),0.0);
    numvec targetvalue = sourcevalue;           // value function to hold the updated values

    numvec residuals(states.size());

    // residual in the policy iteration part
    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    for(i = 0; i < iterations_pi; i++){

        if(print_progress)
            cout << "Policy iteration " << i << "/" << iterations_pi << ":" << endl;

        // this should use move semantics and therefore be very efficient
        swap(targetvalue, sourcevalue);

        prec_t residual_vi = numeric_limits<prec_t>::infinity();

        // update policies
        #pragma omp parallel for
        for(auto s = 0l; s < long(states.size()); s++){
            prec_t newvalue;
            tie(newvalue, policy[s]) = response.policy_update(states[s], s, sourcevalue, discount);
            residuals[s] = abs(sourcevalue[s] - newvalue);
            targetvalue[s] = newvalue;
        }
        residual_pi = *max_element(residuals.cbegin(),residuals.cend());

        if(print_progress) cout << "    Bellman residual: " << residual_pi << endl;

        // the residual is sufficiently small
        if(residual_pi <= maxresidual_pi)
            break;

        if(print_progress) cout << "    Value iteration: " << flush;
        // compute values using value iteration

        for(size_t j = 0; j < iterations_vi &&
               residual_vi > maxresidual_vi_rel * residual_pi;
               j++){
            if(print_progress) cout << "." << flush;

            swap(targetvalue, sourcevalue);

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                prec_t newvalue = response.compute_value(policy[s], states[s], sourcevalue, discount);
                residuals[s] = abs(sourcevalue[s] - newvalue);
                targetvalue[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
        if(print_progress) cout << endl << "    Residual (fixed policy): " << residual_vi << endl << endl;
    }
    auto finish = chrono::steady_clock::now();
    chrono::duration<double> duration = finish-start;
    return Solution<policy_type>(move(targetvalue), move(policy),residual_pi,i,duration.count());
}

// **************************************************************************
// Convenient interface methods
// **************************************************************************


/** 
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the last state to the first one, the states should be ordered
in the temporal order.

@param mdp The MDP to solve
@param discount Discount factor.
@param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
@param policy Partial policy specification. Optimize only actions that are  policy[state] = -1
@param iterations Maximal number of iterations to run
@param maxresidual Stop when the maximal residual falls below this value.


@returns Solution that can be used to compute the total return, or the optimal policy.
*/
template<class SType>
inline DeterministicSolution
solve_vi(const GRMDP<SType>& mdp, prec_t discount,
                        numvec valuefunction=numvec(0), const indvec& policy = indvec(0),
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC)
                        {
   return vi_gs<SType, PlainBellman<SType>>(mdp, discount, move(valuefunction),
            PlainBellman<SType>(policy), iterations, maxresidual);
}


/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
@param type Type of realization of the uncertainty
@param discount Discount factor
@param valuefunction Initial value function
@param policy Partial policy specification. Optimize only actions that are  policy[state] = -1
@param iterations_pi Maximal number of policy iteration steps
@param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
@param iterations_vi Maximal number of inner loop value iterations
@param maxresidual_vi Stop policy evaluation when the policy residual drops below
                            maxresidual_vi * last_policy_residual
@param print_progress Whether to report on progress during the computation
@return Computed (approximate) solution
 */
template<class SType>
inline DeterministicSolution
solve_mpi(const GRMDP<SType>& mdp, prec_t discount,
                const numvec& valuefunction=numvec(0), const indvec& policy = indvec(0),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=0.9,
                bool print_progress=false) {

    return mpi_jac<SType, PlainBellman<SType>>(mdp, discount, valuefunction, PlainBellman<SType>(policy),
                    iterations_pi, maxresidual_pi,
                     iterations_vi, maxresidual_vi, 
                     print_progress);
}

}

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

namespace craam {
/// Main namespace for algorithms that operate on MDPs and RMDPs
namespace algorithms{

using namespace std;
using namespace util::lang;


// *******************************************************
// RegularAction computation methods
// *******************************************************

/**
Computes the average value of the action.

\param action Action for which to compute the value
\param valuefunction State value function to use
\param discount Discount factor
\return Action value
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

\param action Action for which to compute the value
\param valuefunction State value function to use
\param discount Discount factor
\param distribution New distribution. The length must match the number of
            states to which the original transition probabilities are strictly greater than 0.
            The order of states is the same as in the underlying transition.
\return Action value
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

\param action Action for which the value is computed
\param valuefunction Updated value function
\param discount Discount factor
\return Mean value of the action
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

\param action Action for which the value is computed
\param valuefunction Updated value function
\param discount Discount factor
\param distribution Custom distribution that is selected by nature.
\return Value of the action
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

\param state State to compute the value for
\param valuefunction Value function to use for the following states
\param discount Discount factor

\return (Index of best action, value), returns 0 if the state is terminal.
*/
template<class AType>
inline pair<long,prec_t> value_max_state(const SAState<AType>& state, const numvec& valuefunction,
                                     prec_t discount) {
    if(state.is_terminal())
        return make_pair(-1,0.0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;

    for(size_t i = 0; i < state.size(); i++){
        auto const& action = state[i];

        // skip invalid state.get_actions()
        if(!state.is_valid(i)) continue;

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

\param state State to compute the value for
\param valuefunction Value function to use for the following states
\param discount Discount factor

\return Value of state, 0 if it's terminal regardless of the action index
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

\param state State to compute the value for
\param valuefunction Value function to use in computing value of states.
\param discount Discount factor
\param distribution New distribution over states with non-zero nominal probabilities

\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t
value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              long actionid, numvec distribution) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return 0;

    assert(actionid >= 0 && actionid < long(state.size()));

    if(actionid < 0 || actionid >= long(state.size())) throw range_error("invalid actionid: " 
        + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if(!state.is_valid(actionid)) throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount, distribution);
}

/**
Computes the value of a fixed (and valid) action. Performs validity checks.

\param state State to compute the value for
\param valuefunction Value function to use for the following states
\param discount Discount factor
\param action_probabilities A vector of probabilities whose indecies correspond to the action id

\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t value_stochastic_state(const SAState<AType>& state, numvec const& valuefunction,
                              prec_t discount, prob_list_t action_probabilities) {
    prec_t return_value = 0;
    for ( int i = 0; i < action_probabilities.size(); i++ ) {
        if ( state.action_count() > i )
            return_value += action_probabilities[i] * value_fix_state(state, valuefunction, discount, i);
        else
            break;
    }
    return return_value;
}

// *******************************************************
// RMDP computation methods
// *******************************************************

/** A set of values that represent a solution to a plain MDP.  */
struct Solution {
    /// Value function
    numvec valuefunction;
    /// index of the action to take for each states
    indvec policy;
    /// Bellman residual of the computation
    prec_t residual;
    /// Number of iterations taken
    long iterations;

    Solution(): valuefunction(0), policy(0), residual(-1),iterations(-1) {}

    /// Empty solution for a problem with statecount states
    Solution(size_t statecount): valuefunction(statecount, 0.0), policy(statecount, -1), residual(-1),iterations(-1) {}

    /// Empty solution for a problem with a given value function and policy
    Solution(numvec valuefunction, indvec policy, prec_t residual = -1, long iterations = -1) :
        valuefunction(move(valuefunction)), policy(move(policy)), residual(residual), iterations(iterations) {}

    /**
    Computes the total return of the solution given the initial
    distribution.
    \param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const{
        if(initial.max_index() >= (long) valuefunction.size()) throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};


/** A set of values that represent a stochastic solution to a plain MDP.  */
struct StochasticSolution {
    /// Value function
    numvec valuefunction;
    /// index of the action to take for each states
    prob_matrix_t policy;
    /// Bellman residual of the computation
    prec_t residual;
    /// Number of iterations taken
    long iterations;

    StochasticSolution(): valuefunction(0), policy(0), residual(-1),iterations(-1) {}

    /// Empty solution for a problem with statecount states
    StochasticSolution(size_t statecount): valuefunction(statecount, 0.0), policy(statecount, prob_list_t(0,0)), residual(-1),iterations(-1) {}

    /// Empty solution for a problem with a given value function and policy
    StochasticSolution(numvec valuefunction, prob_matrix_t policy, prec_t residual = -1, long iterations = -1) :
        valuefunction(move(valuefunction)), policy(move(policy)), residual(residual), iterations(iterations) {}

    /**
    Computes the total return of the solution given the initial
    distribution.
    \param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const{
        if(initial.max_index() >= (long) valuefunction.size()) throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};


// **************************************************************************
// Helper classes to handle computing of the best response
// **************************************************************************

/**
A Bellman update class for solving regular Markov decision processes. This class abstracts
away from particular model properties and the goal is to be able to plug it in into value or
policy iteration methods for MDPs.

Many of the methods are parametrized by the type of the state.
*/
class PlainBellman{
protected:
    /// Partial policy specification (action -1 is ignored and optimized)
    indvec initial_policy;

public:
    /// Provides the type of the solution for the consumption of methods that call this value
    using solution_type = Solution;

    /// Constructs the update with no constraints on the initial policy
    PlainBellman() : initial_policy(0) {}

    /** A partial policy that can be used to fix some actions
     *  @param policy policy[s] = -1 means that the action should be optimized in the state
     *                policy of length 0 means that all actions will be optimized
     */
    PlainBellman(indvec policy) : initial_policy(move(policy)) {}

    /**
     * @brief Constructs a solution for the
     * @param statecount
     * @param valuefunction
     * @return
     */
    Solution new_solution(size_t statecount, numvec valuefunction) const {
        process_valuefunction(statecount, valuefunction);
        Solution solution =  Solution(move(valuefunction), process_policy(statecount));
        return solution;
    }

    /**
     *  Computes the Bellman update and updates the internal policy to the best response.
     *  This function does *not* update the value function.
     *  \returns New value for the state
     */
    template<class SType>
    prec_t policy_update(solution_type& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{
        assert(stateid < long(solution.policy.size()));

        prec_t newvalue;
        // check whether this state should only be evaluated
        if(initial_policy.empty() || initial_policy[stateid] < 0){    // optimizing
            tie(solution.policy[stateid], newvalue) = value_max_state(state, valuefunction, discount);
        }else{// fixed-action, do not copy
            return value_fix_state(state, valuefunction, discount, initial_policy[stateid]);
        }
        return newvalue;
    }

    /**
     *  Computes value function update using the current policy
     * \returns New value for the state
     */
    template<class SType>
    prec_t compute_value(const solution_type& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{

        return value_fix_state(state, valuefunction, discount, solution.policy[stateid]);
    }

protected:

    /**
     * Resizes value function to the proper length when it is empty (size == 0)
     * @param statecount Number of states in the MDP
     * @param valuefunction Value function
     */
    void process_valuefunction(size_t statecount, numvec& valuefunction) const{
        // check if the value function is a correct size, and if it is length 0
        // then creates an appropriate size
        if(!valuefunction.empty()){
            if(valuefunction.size() != statecount) throw invalid_argument("Incorrect dimensions of value function.");
        }else{
            valuefunction.assign(statecount, 0.0);
        }
        assert(valuefunction.size() == statecount);
    }

    /**
     * @brief process_policy Initializes policy to match the number of states.
     *
     * Does not modify internal policy object, but returns the new policy.
     *
     * @param statecount Number of states in the MDP
     * @return Policy of the appropriate size.
     */
    indvec process_policy(size_t statecount) const {
        // check the dimensions of the policy
        if(!initial_policy.empty()){
            if(initial_policy.size() != statecount) throw invalid_argument("Incorrect dimensions of policy function.");
            return initial_policy;
        }else{
            return indvec(statecount, -1);
        }
    }
};


/**
A Bellman update class for solving regular Markov decision processes. This class abstracts
away from particular model properties and the goal is to be able to plug it in into value or
policy iteration methods for MDPs.

Many of the methods are parametrized by the type of the state.
*/
class StochasticBellman {
protected:
    /// Partial policy specification (action -1 is ignored and optimized)
    prob_matrix_t initial_policy;
    int action_count;

public:
    /// Provides the type of the solution for the consumption of methods that call this value
    using solution_type = StochasticSolution;

    /// Constructs the update with no constraints on the initial policy
    StochasticBellman(int action_count) : action_count(action_count), initial_policy(0) {}

    /** A partial policy that can be used to fix some actions
     *  @param policy policy[s] = -1 means that the action should be optimized in the state
     *                policy of length 0 means that all actions will be optimized
     */
    StochasticBellman(int action_count, prob_matrix_t policy) : action_count(action_count), initial_policy(move(policy)) {}

    /**
     * @brief Constructs a solution for the
     * @param statecount
     * @param valuefunction
     * @return
     */
    StochasticSolution new_solution(size_t statecount, numvec valuefunction) const {
        process_valuefunction(statecount, valuefunction);
        StochasticSolution solution =  StochasticSolution(move(valuefunction), process_policy(statecount, action_count));
        return solution;
    }

    /**
     *  Computes the Bellman update and updates the internal policy to the best response.
     *  This function does *not* update the value function.
     *  \returns New value for the state
     */
    template<class SType>
    prec_t policy_update(solution_type& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const {
        assert(stateid < long(solution.policy.size()));

        prec_t newvalue;
        // check whether this state should only be evaluated
        if(initial_policy.empty() || initial_policy[stateid].empty() || sum_vector(initial_policy[stateid]) < 0){    // optimizing
            pair<long,prec_t> results = value_max_state(state, valuefunction, discount);
            newvalue = results.second;
            for ( int i = 0; i < solution.policy[stateid].size(); i++ )
                solution.policy[stateid][i] = 0;
            solution.policy[stateid][results.first] = 1;
        }else{// fixed-action, do not copy
            return value_stochastic_state(state, valuefunction, discount, initial_policy[stateid]);
        }
        return newvalue;
    }

    /**
     *  Computes value function update using the current policy
     * \returns New value for the state
     */
    template<class SType>
    prec_t compute_value(const solution_type& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{

        return value_stochastic_state(state, valuefunction, discount, solution.policy[stateid]);
    }

protected:

    /**
     * Resizes value function to the proper length when it is empty (size == 0)
     * @param statecount Number of states in the MDP
     * @param valuefunction Value function
     */
    void process_valuefunction(size_t statecount, numvec& valuefunction) const{
        // check if the value function is a correct size, and if it is length 0
        // then creates an appropriate size
        if(!valuefunction.empty()){
            if(valuefunction.size() != statecount) throw invalid_argument("Incorrect dimensions of value function.");
        }else{
            valuefunction.assign(statecount, 0.0);
        }
        assert(valuefunction.size() == statecount);
    }

    /**
     * @brief process_policy Initializes policy to match the number of states.
     *
     * Does not modify internal policy object, but returns the new policy.
     *
     * @param statecount Number of states in the MDP
     * @return Policy of the appropriate size.
     */
    prob_matrix_t process_policy(size_t state_count, size_t action_count) const {
       // check the dimensions of the policy
        if(!initial_policy.empty()){
            if(initial_policy.size() != state_count) throw invalid_argument("Incorrect dimensions of policy function.");
            return initial_policy;
        }else{
            prob_matrix_t new_policy(state_count, prob_list_t(action_count,0));
            return new_policy;
        }
    }

private:
    double sum_vector(const prob_list_t &list) const {
        double total = 0;
        for (int i = 0; i < list.size(); i++)
            total += list[i];
        return total;
    }
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

\param mdp The mdp to solve
\param discount Discount factor.
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
\param response Using PolicyResponce allows to specify a partial policy. Only the actions that 
                not provided by the partial policy are included in the optimization. 
                Using a class of a different types enables computing other objectives,
                such as robust or risk averse ones. 
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.


\returns Solution that can be used to compute the total return, or the optimal policy.
 */
template<class SType, class ResponseType = PlainBellman>
inline auto vi_gs(const GRMDP<SType>& mdp, prec_t discount,
                        numvec valuefunction=numvec(0), const ResponseType& response = PlainBellman(),
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC)
                        {

    const auto& states = mdp.get_states();
    typename ResponseType::solution_type solution =
            response.new_solution(states.size(), move(valuefunction));
    // just quit if there are no states
    if( mdp.state_count() == 0) return solution;

    // initialize values
    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;   // iterations defined outside to make them reportable

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            prec_t newvalue = response.policy_update(solution, states[s], s, solution.valuefunction, discount);

            residual = max(residual, abs(solution.valuefunction[s] - newvalue));
            solution.valuefunction[s] = newvalue;
        }
    }
    solution.residual = residual;
    solution.iterations = i;
    return solution;
}


/**
Modified policy iteration using Jacobi value iteration in the inner loop. See solve_mpi for 
a simplified interface.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param valuefunction Initial value function
\param response Using PolicyResponce allows to specify a partial policy. Only the actions that 
                not provided by the partial policy are included in the optimization. 
                Using a class of a different types enables computing other objectives,
                such as robust or risk averse ones. 
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param print_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType, class ResponseType = PlainBellman>
inline auto mpi_jac(const GRMDP<SType>& mdp, prec_t discount,
                const numvec& valuefunction=numvec(0), const ResponseType& response = PlainBellman(),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {

    const auto& states = mdp.get_states();
    typename ResponseType::solution_type solution =
            response.new_solution(states.size(), move(valuefunction));

    // just quit if there are no states
    if( mdp.state_count() == 0) return solution;

    numvec sourcevalue = solution.valuefunction;   // value function to compute the update
    numvec targetvalue = sourcevalue;              // value function to hold the updated values

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
            prec_t newvalue = response.policy_update(solution, states[s], s, sourcevalue, discount);
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

        for(size_t j = 0; j < iterations_vi && residual_vi > maxresidual_vi; j++){
            if(print_progress) cout << "." << flush;

            swap(targetvalue, sourcevalue);

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                prec_t newvalue = response.compute_value(solution, states[s], s, sourcevalue, discount);
                residuals[s] = abs(sourcevalue[s] - newvalue);
                targetvalue[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
        if(print_progress) cout << endl << "    Residual (fixed policy): " << residual_vi << endl << endl;
    }
    solution.valuefunction = move(targetvalue);
    solution.residual = residual_pi;
    solution.iterations = i;
    return solution;
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

\param mdp The MDP to solve
\param discount Discount factor.
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
\param policy Partial policy specification. Optimize only actions that are  policy[state] = -1
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.


\returns Solution that can be used to compute the total return, or the optimal policy.
*/
template<class SType>
inline auto solve_vi(const GRMDP<SType>& mdp, prec_t discount,
                        numvec valuefunction=numvec(0), const indvec& policy = indvec(0),
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC)
                        {
   return vi_gs<SType, PlainBellman>(mdp, discount, move(valuefunction),
            PlainBellman(policy), iterations, maxresidual);
}


/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param valuefunction Initial value function
\param policy Partial policy specification. Optimize only actions that are  policy[state] = -1
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param print_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType>
inline auto solve_mpi(const GRMDP<SType>& mdp, prec_t discount,
                const numvec& valuefunction=numvec(0), const indvec& policy = indvec(0),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {

    return mpi_jac<SType, PlainBellman>(mdp, discount, valuefunction, PlainBellman(policy),
                    iterations_pi, maxresidual_pi,
                     iterations_vi, maxresidual_vi, 
                     print_progress);
}

template<class SType>
inline auto solve_mpi_stochastic(const GRMDP<SType>& mdp, prec_t discount,
                const numvec& valuefunction=numvec(0), const prob_matrix_t& policy = prob_matrix_t(0),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {
    int action_count = mdp.action_count();

    return mpi_jac<SType, StochasticBellman>(mdp, discount, valuefunction, StochasticBellman(action_count, policy),
                    iterations_pi, maxresidual_pi,
                     iterations_vi, maxresidual_vi,
                     print_progress);
}

}}

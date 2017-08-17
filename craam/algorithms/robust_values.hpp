/**
Robust MDP methods for computing value functions.
*/
#pragma once

#include "../RMDP.hpp"
#include "values.hpp"
#include <functional>
#include <type_traits>
#include "../cpp11-range-master/range.hpp"

namespace craam { namespace algorithms{

using namespace std;
using namespace util::lang;

// *******************************************************
// Nature definitions 
// *******************************************************

/**
Function representing constraints on nature. The function computes
the best response of nature and can be used in value iteration.

This function represents a nature which computes (in general) a randomized
policy (response). If the response is always deterministic, it may be better
to define and use a nature that computes and uses a deterministic response.

The parameters are the q-values v, the reference distribution p, and the threshold.
The function returns the worst-case solution and the objective value. The threshold can
be used to determine the desired robustness of the solution.
*/
template<class T>
using NatureResponse = vec_scal_t (*)(numvec const& v, numvec const& p, T threshold);

/**
Represents an instance of nature that can be used to directly compute the response.
*/
template<class T>
using NatureInstance = pair<NatureResponse<T>, T>;


/// L1 robust response
inline vec_scal_t robust_l1(const numvec& v, const numvec& p, prec_t threshold){
    assert(v.size() == p.size());
    return worstcase_l1(v,p,threshold);
}

/// L1 optimistic response
inline vec_scal_t optimistic_l1(const numvec& v, const numvec& p, prec_t threshold){
    assert(v.size() == p.size());
    //TODO: this could be faster without copying the vector and just modifying the function
    numvec minusv(v.size());
    transform(begin(v), end(v), begin(minusv), negate<prec_t>());
    auto&& result = worstcase_l1(minusv,p,threshold);
    return make_pair(result.first, -result.second);
}

/// worst outcome, threshold is ignored
template<class T>
inline vec_scal_t robust_unbounded(const numvec& v, const numvec& p, T){
    assert(v.size() == p.size());
    numvec dist(v.size(),0.0);
    long index = min_element(begin(v), end(v)) - begin(v);
    dist[index] = 1;
    return make_pair(dist,v[index]);
}

/// best outcome, threshold is ignored
template<class T>
inline vec_scal_t optimistic_unbounded(const numvec& v, const numvec& p, T){
    assert(v.size() == p.size());
    numvec dist(v.size(),0.0);
    long index = max_element(begin(v), end(v)) - begin(v);
    dist[index] = 1;
    return make_pair(dist,v[index]);
}

// *******************************************************
// RegularAction computation methods
// *******************************************************


/**
Computes an ambiguous value (e.g. robust) of the action, depending on the type
of nature that is provided.

\param action Action for which to compute the value
\param valuefunction State value function to use
\param discount Discount factor
\param nature Method used to compute the response of nature.
*/
template<class T>
inline vec_scal_t value_action(const RegularAction& action, const numvec& valuefunction,
                        prec_t discount, const NatureInstance<T>& nature){

    const numvec& rewards = action.get_outcome().get_rewards();
    const indvec& nonzero_indices = action.get_outcome().get_indices();

    numvec qvalues(rewards.size()); // values for individual states - used by nature.

    #pragma omp simd
    for(size_t i = 0; i < rewards.size(); i++){
        qvalues[i] = rewards[i] + discount * valuefunction[nonzero_indices[i]];
    }

    return nature.first(qvalues, action.get_outcome().get_probabilities(), nature.second);
}



// *******************************************************
// WeightedOutcomeAction computation methods
// *******************************************************

/**
Computes the maximal outcome distribution constraints on the nature's distribution.
Does not work when the number of outcomes is zero.

\param action Action for which the value is computed
\param valuefunction Value function reference
\param discount Discount factor
\param nature Method used to compute the response of nature.

\return Outcome distribution and the mean value for the choice of the nature
 */
template<class T>
inline vec_scal_t value_action(const WeightedOutcomeAction& action, numvec const& valuefunction,
                                prec_t discount, const NatureInstance<T> nature) {

    assert(action.get_distribution().size() == action.get_outcomes().size());

    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");

    numvec outcomevalues(action.size());
    for(size_t i = 0; i < action.size(); i++)
        outcomevalues[i] = action[i].value(valuefunction, discount);

    return nature.first(outcomevalues, action.get_distribution(), nature.second);
}


// *******************************************************
// State computation methods
// *******************************************************


/**
Computes the value of a fixed action and any response of nature.

\param state State to compute the value for
\param valuefunction Value function to use in computing value of states.
\param discount Discount factor
\param nature Instance of a nature optimizer

\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType, class T>
inline vec_scal_t
value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              long actionid, const NatureInstance<T>& nature) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return make_pair(numvec(0),0);

    assert(actionid >= 0 && actionid < long(state.size()));

    if(actionid < 0 || actionid >= (long) state.size()) throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if(!state.is_valid(actionid)) throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount, nature);
}



/**
Finds the greedy action and its value for the given value function.
This function assumes a robust or optimistic response by nature depending on the provided
ambiguity.

When there are no actions, the state is assumed to be terminal and the return is 0.

\param state State to compute the value for
\param valuefunction Value function to use in computing value of states.
\param discount Discount factor
\param natures Method used to compute the response of nature; one for each action available in the state.

\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<typename AType, typename T>
inline ind_vec_scal_t
value_max_state(const SAState<AType>& state, const numvec& valuefunction,
                prec_t discount, const vector<NatureInstance<T>>& natures) {

    // can finish immediately when the state is terminal
    if(state.is_terminal())
        return make_tuple(-1,numvec(),0);

    // make sure that the number of natures is the same as the number of actions
    assert(natures.size() == state.size());

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();

    long result = -1;
    numvec result_outcome;

    for(size_t i = 0; i < state.get_actions().size(); i++){
        const auto& action = state[i];

        // skip invalid state.get_actions()
        if(!state.is_valid(i)) continue;

        auto value = value_action(action, valuefunction, discount, natures[i]);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = move(value.first);
        }
    }

    // if the result has not been changed, that means that all actions are invalid
    if(result == -1)
        throw invalid_argument("all actions are invalid.");

    return make_tuple(result,result_outcome,maxvalue);
}

/**
A wrapper that assumes that the natures is the same across all the actions
*/
template<typename AType, typename T>
inline ind_vec_scal_t
value_max_state(const SAState<AType>& state, const numvec& valuefunction,
                prec_t discount, const NatureInstance<T>& nature) {
    return value_max_state(state, valuefunction, discount, vector<NatureInstance<T>>(state.action_count(), nature));
}

// **************************************************************************
// Helper classes to handle computing of the best response
// **************************************************************************

/** A robust solution to a robust or regular MDP.  */
struct SolutionRobust : public Solution {
    /// Randomized policy of nature, probabilities only for states that have
    /// non-zero probability in the MDP model (or for outcomes)
    vector<numvec> natpolicy;

    /// Empty SolutionRobust
    SolutionRobust() : Solution(), natpolicy(0) {}

    /// Empty SolutionRobust for a problem with statecount states
    SolutionRobust(size_t statecount): Solution(statecount), natpolicy(statecount, numvec(0)) {}

    /// Empty SolutionRobust for a problem with policy and value function
    SolutionRobust(numvec valuefunction, indvec policy):
            Solution(move(valuefunction), move(policy)),
            natpolicy(this->valuefunction.size(), numvec(0)) {}

    SolutionRobust(numvec valuefunction, indvec policy,
             vector<numvec> natpolicy, prec_t residual = -1, long iterations = -1) :
        Solution(move(valuefunction), move(policy), residual, iterations),
        natpolicy(move(natpolicy)) {}
};
/**
The class abstracts some operations of value / policy iteration in order to generalize to
various types of robust MDPs. It can be used in place of response in mpi_jac or vi_gs to 
solve robust MDP objectives.
*/
template<class T>
class PolicyNature : public PolicyDeterministic {
public:
    using solution_type = SolutionRobust;

    /** Specification of natures response (the function that nature computes).
    The outer list is for each state and the inner list is for each action in the state. */
    vector<vector<NatureInstance<T>>> natspec;

    /**
    Constructs the object from a policy and a specification of nature. Action are optimized
    only in states in which policy is -1 (or < 0)
    @param policy Index of the action to take for each state
    @param natspec Responce of nature for each state and each action
    */
    PolicyNature(indvec policy, vector<vector<NatureInstance<T>>> natspec):
        PolicyDeterministic(move(policy)), natspec(move(natspec)) {}

    /**
    Constructs the object from a specification of nature. No decision maker's
    policy is provided.
    @param natspec Responce of nature for each state and each action
    */
    PolicyNature(vector<vector<NatureInstance<T>>> natspec):
        PolicyDeterministic(indvec(0)), natspec(move(natspec)) {}

    /// Constructs a new robust solution
    SolutionRobust new_solution(size_t statecount, numvec valuefunction) const {

        //if(valuefunction.size() != statecount)
         //   throw invalid_argument("Size of value function specification does not match the number of states.");

        process_valuefunction(statecount, valuefunction);
        SolutionRobust solution =  SolutionRobust(move(valuefunction), process_policy(statecount));
        return solution;
    }


    /**
    Computes the Bellman update and updates the action in the solution to the best response
    It does not update the value function in the solution.
    @param solution Solution to update
    @param state State for which to compute the Bellman update
    @param stateid  Index of the state
    @param valuefunction Value function
    @param discount Discount factor
    @returns New value for the state
     */
    template<class SType>
    prec_t update_solution(SolutionRobust& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{
        prec_t newvalue = 0;
        // check whether this state should only be evaluated or also optimized
        // optimizing action
        if(policy.empty() || policy[stateid] < 0){
            tie(solution.policy[stateid], solution.natpolicy[stateid], newvalue) =
                    value_max_state(state, valuefunction, discount, natspec[stateid]);
        }
        // fixed-action, do not copy
        else{
            prec_t newvalue;
            const long actionid = policy[stateid];
            tie(solution.natpolicy[stateid], newvalue) = value_fix_state(state, valuefunction, discount, actionid, natspec[stateid][actionid]);
        }
        return newvalue;
    }

    /**
    Computes value function using the provided policy. Used in policy evaluation.
    @param solution Solution used to infer the current policy
    @param state State for which to compute the Bellman update
    @param stateid Index of the state
    @param valuefunction Value function
    @param discount Discount factor
    @returns New value for the state
    */
    template<class SType>
    prec_t update_value(const SolutionRobust& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{
        return value_fix_state(state, valuefunction, discount, solution.policy[stateid],
                solution.natpolicy[stateid]);
    }
};

/**
A helper function that simply copies a nature specification across all states and actions
*/
template<class T>
PolicyNature<T> uniform_nature(const MDP& m, NatureResponse<T> nature,
                            T threshold){
    vector<vector<NatureInstance<T>>> natures(m.state_count());

    for(size_t stateid = 0; stateid < m.state_count(); stateid++){
        natures[stateid] = vector<NatureInstance<T>>(m[stateid].action_count(), make_pair(nature, threshold));
    }
    return PolicyNature<T>(natures);
}

/**
A helper function that simply copies a nature specification across all states and actions
*/
template<class T>
PolicyNature<T> uniform_nature(const RMDP& m, NatureResponse<T> nature,
                            T threshold){
    vector<vector<NatureInstance<T>>> natures(m.state_count());

    for(size_t stateid = 0; stateid < m.state_count(); stateid++){
        natures[stateid] = vector<NatureInstance<T>>(m[stateid].action_count(), make_pair(nature, threshold));
    }
    return PolicyNature<T>(natures);
}


namespace internal{

/// Zips two vectors
template <class T1, class T2>
vector<pair<T1,T2>> zip(const vector<T1>& v1, const vector<T2>& v2){

    assert(v1.size() == v2.size());
    vector<pair<T1,T2>> result(v1.size());
    for(size_t i=0; i< v1.size(); i++){
        result[i] = make_pair(v1[i], v2[i]);
    }
    return result;
}

/// Zips two vectors of vectors
template <class T1, class T2>
vector<vector<pair<T1,T2>>> zip(const vector<vector<T1>>& v1, const vector<vector<T2>>& v2){

    assert(v1.size() == v2.size());
    vector<vector<pair<T1,T2>>> result(v1.size());
    for(size_t i=0; i< v1.size(); i++){
        result[i] = zip(v1[i], v2[i]);
    }
    return result;
}

/// Zips a single value with a vector
template <class T1, class T2>
vector<pair<T1,T2>> zip(const T1& v1, const vector<T2>& v2){

    vector<pair<T1,T2>> result(v2.size());
    for(size_t i=0; i< v2.size(); i++){
        result[i] = make_pair(v1, v2[i]);
    }
    return result;
}

/// Zips a single value with a vector of vectors
template <class T1, class T2>
vector<vector<pair<T1,T2>>> zip(const T1& v1, const vector<vector<T2>>& v2){

    vector<vector<pair<T1,T2>>> result(v2.size());
    for(size_t i=0; i< v2.size(); i++){
        result[i] = zip(v1, v2[i]);
    }
    return result;
}



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

This is a simplified method interface. Use vi_gs with PolicyNature for full functionality.

\param mdp The MDP to solve
\param discount Discount factor.
\param nature Response of nature, the function is the same for all states and actions.
\param thresholds Parameters passed to nature response functions.
                    One value per state and then one value per action.
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
\param policy Partial policy specification. Optimize only actions that are  policy[state] = -1
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.


\returns Solution that can be used to compute the total return, or the optimal policy.
*/
template<class SType, class T = prec_t >
inline auto rsolve_vi(const GRMDP<SType>& mdp, prec_t discount,
                        const NatureResponse<T>& nature, const vector<vector<T>>& thresholds,
                        numvec valuefunction, const indvec& policy = indvec(0),
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC)
    {
    // check that the provided sizes are OK
    //assert(nature.size() == thresholds.size());
    //assert(nature.size() == mdp.state_count());
    for(size_t t = 0; t < thresholds.size(); t++)
        assert(thresholds[t].size() == mdp[t].size());

    return vi_gs<SType, PolicyNature<T>>(mdp, discount, move(valuefunction), 
            PolicyNature<T>(policy,internal::zip(nature,thresholds)), 
            iterations, maxresidual);
}


/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

This is a simplified method interface. Use mpi_jac with PolicyNature for full functionality.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param nature Response of nature; the same value is used for all states and actions.
\param thresholds Parameters passed to nature response functions. One value per state (outer) and action (inner).
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
template<class SType, class T = prec_t>
inline auto rsolve_mpi(const GRMDP<SType>& mdp, prec_t discount,
                const NatureResponse<T>& nature, const vector<vector<T>>& thresholds,
                const numvec& valuefunction, const indvec& policy = indvec(0),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {

    assert(nature.size() == thresholds.size());
    assert(nature.size() == mdp.state_count());

    return mpi_jac<SType, PolicyNature<T>>(mdp, discount, valuefunction, 
                    PolicyNature<T>(policy,internal::zip(nature,thresholds)), 
                    iterations_pi, maxresidual_pi,
                    iterations_vi, maxresidual_vi, 
                    print_progress);
}

// ******************************************************************
// Helper functions for a Python or R interface
// ******************************************************************

/**
Converts a string representation of nature response to the appropriate nature response call.
This function is useful when the code is used within a python or R libraries. The values values
correspond to the function definitions, and ones that are currently supported are:

- robust_unbounded
- optimistic_unbounded
- robust_l1
- optimistic_l1
*/
inline NatureResponse<prec_t> string_to_nature(string nature){
    if(nature == "robust_unbounded") return robust_unbounded;
    if(nature == "optimistic_unbounded") return optimistic_unbounded;
    if(nature == "robust_l1") return robust_l1;
    if(nature == "optimistic_l1") return optimistic_l1;
    throw invalid_argument("Unknown nature.");
}

/**
Creates a vector of vector of state thresholds. This function can be used to provide the parameters
to rsolve_mpi and rsolve_vi.

All state-action combinations should be provided. Values that are not provided are undefined.

@param states Indexes of states. The values are 0-based.
@param actions Indexes of actions. The values are 0-based.
@param values Values that correspond to the states and actions.
*/
template<class T>
vector<vector<T>> pack_thresholds(const indvec& states, const indvec& actions, const vector<T>& values){
    assert(states.size() == actions.size());
    assert(states.size() == values.size());

    vector<vector<T>> result(*max_element(states.cbegin(), states.cend())+1);

    for(size_t i = 0; i < states.size(); i++){
        auto& statelist = result[states[i]];
        if(statelist.size() <= actions[i])
            statelist.resize(actions[i]+1);
        statelist[actions[i]] = values[i];
    }
    return result;
}

}}

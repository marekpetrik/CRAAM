#pragma once

#include "../RMDP.hpp"
#include <functional>
#include <type_traits>
#include "../cpp11-range-master/range.hpp"

namespace craam {namespace algorithms{

using namespace std;
using namespace util::lang;


/**
Function representing the constraints on nature. The function computes
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
inline vec_scal_t robust_unbounded(const numvec& v, const numvec& p, T threshold){
    assert(v.size() == p.size());
    numvec dist(v.size(),0.0);
    long index = min_element(begin(v), end(v)) - begin(v);
    dist[index] = 1;
    return make_pair(dist,v[index]);
}

/// best outcome, threshold is ignored
template<class T>
inline vec_scal_t optimistic_unbounded(const numvec& v, const numvec& p, T threshold){
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
        if(!action.is_valid()) continue;

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
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume invalid state.get_actions()
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");

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

    assert(actionid >= 0 && actionid < state.size());

    if(actionid < 0 || actionid >= (long) state.size()) throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount, distribution);
}

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

    assert(actionid >= 0 && actionid < state.size());

    if(actionid < 0 || actionid >= (long) state.size()) throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );

    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");

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
\param nature Method used to compute the response of nature.

\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<typename AType, typename T>
inline ind_vec_scal_t
value_max_state(const SAState<AType>& state, const numvec& valuefunction,
                prec_t discount, const NatureInstance<T>& nature) {

    if(state.is_terminal())
        return make_tuple(-1,numvec(),0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();

    long result = -1;
    numvec result_outcome;

    for(size_t i = 0; i < state.get_actions().size(); i++){
        const auto& action = state.get_actions()[i];

        // skip invalid state.get_actions()
        if(!action.is_valid()) continue;

        auto value = value_action(action, valuefunction, discount, nature);
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

// *******************************************************
// RMDP computation methods
// *******************************************************

/** A solution to a plain MDP.  */
struct Solution {
    /// Value function
    numvec valuefunction;
    /// index of the action to take for each states
    indvec policy;
    /// Bellman residual of the computation
    prec_t residual;
    /// Number of iterations taken
    long iterations;

    Solution(): valuefunction(0), policy(0), residual(-1),iterations(-1) {};

    /// Empty solution for a problem with statecount states
    Solution(size_t statecount): valuefunction(statecount, 0.0), policy(statecount, -1), residual(-1),iterations(-1) {};

    /// Empty solution for a problem with statecount states
    Solution(size_t statecount, numvec valuefunction, indvec policy):
                valuefunction(move(valuefunction)),
                policy(move(policy)),
                residual(-1),iterations(-1) {};

    Solution(numvec valuefunction, indvec policy, prec_t residual = -1, long iterations = -1) :
        valuefunction(move(valuefunction)), policy(move(policy)), residual(residual), iterations(iterations) {};

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
/** A robust solution to a robust or regular MDP.  */
struct RSolution : public Solution {
    /// Randomized policy of nature, probabilities only for states that have
    /// non-zero probability in the MDP model (or for outcomes)
    vector<numvec> natpolicy;

    /// Empty RSolution
    RSolution() : Solution(), natpolicy(0) {};

    /// Empty RSolution for a problem with statecount states
    RSolution(size_t statecount): Solution(statecount), natpolicy(statecount, numvec(0)) {};

    /// Empty RSolution for a problem with statecount states
    RSolution(size_t statecount, numvec valuefunction, indvec policy):
            Solution(statecount, move(valuefunction), move(policy)),
            natpolicy(statecount, numvec(0)) {};

    RSolution(numvec valuefunction, indvec policy,
             vector<numvec> natpolicy, prec_t residual = -1, long iterations = -1) :
        Solution(move(valuefunction), move(policy), residual, iterations),
        natpolicy(move(natpolicy)) {};
};


// **************************************************************************
// Helper classes to handle computing of the best response
// **************************************************************************

/*
Regular solution to an MDP

Field policy Ignored when size is 0. Otherwise a partial policy. Actions are optimized only in
                 states in which policy = -1, otherwise a fixed value is used.
*/
class PolicyDeterministic{
public:
    using solution_type = Solution;

    /// Partial policy specification (action -1 is ignored and optimized)
    indvec policy;

    /// All actions will be optimized
    PolicyDeterministic() : policy(0) {};

    /// A partial policy that can be used to fix some actions
    /// policy[s] = -1 means that the action should be optimized in the state
    /// policy of length 0 means that all actions will be optimized
    PolicyDeterministic(indvec policy) : policy(move(policy)) {};

    Solution new_solution(size_t statecount, numvec valuefunction) const {
        process_valuefunction(statecount, valuefunction);
        assert(valuefunction.size() == statecount);
        Solution solution =  Solution(statecount,move(valuefunction), process_policy(statecount));
        return solution;
    }

    /// Computed the Bellman update and updates the solution to the best response
    /// It does not update the value function
    /// \returns New value for the state
    template<class SType>
    prec_t update_solution(Solution& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{
        assert(stateid < solution.policy.size());

        prec_t newvalue;
        // check whether this state should only be evaluated
        if(policy.empty() || policy[stateid] < 0){    // optimizing
            tie(solution.policy[stateid], newvalue) = value_max_state(state, valuefunction, discount);
        }else{// fixed-action, do not copy
            return value_fix_state(state, valuefunction, discount, policy[stateid]);
        }
        return newvalue;
    }

    /// Computes a fixed Bellman update using the current solution policy
    /// \returns New value for the state
    template<class SType>
    prec_t update_value(const Solution& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{

        return value_fix_state(state, valuefunction, discount, solution.policy[stateid]);
    }
protected:
    void process_valuefunction(size_t statecount, numvec& valuefunction) const{
        // check if the value function is a correct size, and if it is length 0
        // then creates an appropriate size
        if(!valuefunction.empty()){
            if(valuefunction.size() != statecount) throw invalid_argument("Incorrect dimensions of value function.");
        }else{
            valuefunction.assign(statecount, 0.0);
        }
    }
    indvec process_policy(size_t statecount) const {
        // check the dimensions of the policy
        if(!policy.empty()){
            if(policy.size() != statecount) throw invalid_argument("Incorrect dimensions of policy function.");
            return policy;
        }else{
            return indvec(statecount, -1);
        }
    }
};

/**
Robust solution to an MDP

The class abstracts some operations of value / policy iteration in order to generalize to
various types of robust MDPs.
*/
template<class T>
class PolicyNature : public PolicyDeterministic {
public:
    using solution_type = RSolution;

    /// Specification of natures response (the function that nature computes, could be different for each state)
    vector<NatureInstance<T>> natspec;

    /// Constructs the object from a policy and a specification of nature
    PolicyNature(indvec policy, vector<NatureInstance<T>> natspec):
        PolicyDeterministic(move(policy)), natspec(move(natspec)) {};

    /// Constructs the object from a policy and a specification of nature
    PolicyNature(vector<NatureInstance<T>> natspec):
        PolicyDeterministic(indvec(0)), natspec(move(natspec)) {};

    /// Constructs a new robust solution
    RSolution new_solution(size_t statecount, numvec valuefunction) const {
        if(natspec.size() != statecount)
            throw invalid_argument("Size of nature specification does not match the number of states.");

        process_valuefunction(statecount, valuefunction);
        RSolution solution =  RSolution(statecount,move(valuefunction),
                                process_policy(statecount));
        return solution;
    }

    /// Computed the Bellman update and updates the solution to the best response
    /// It does not update the value function
    /// \returns New value for the state
    template<class SType>
    prec_t update_solution(RSolution& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{

        prec_t newvalue;
        // check whether this state should only be evaluated or also optimized
        if(policy.empty() || policy[stateid] < 0){    // optimizing
            tie(solution.policy[stateid], solution.natpolicy[stateid], newvalue) = value_max_state(state, valuefunction, discount, natspec[stateid]);
        }else{// fixed-action, do not copy
            prec_t newvalue;
            tie(solution.natpolicy[stateid], newvalue) = value_fix_state(state, valuefunction, discount, policy[stateid], natspec[stateid]);
        }
        return newvalue;
    }

    /// Computes a fixed Bellman update using the current solution policy
    /// \returns New value for the state
    template<class SType>
    prec_t update_value(const RSolution& solution, const SType& state, long stateid,
                            const numvec& valuefunction, prec_t discount) const{

        return value_fix_state(state, valuefunction, discount, solution.policy[stateid],
                solution.natpolicy[stateid]);
    }

};


// **************************************************************************
// Main solution methods
// **************************************************************************

/**
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the last state to the first one, the states should be ordered
in the temporal order.

\param mdp The mdp to solve
\param discount Discount factor.
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use
                    all zeros when not provided. Ignored when size is 0.
\param response Determines the type of solution method. Allows for customized algorithms
                that can solve various forms of robustness, and risk aversion
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.


\returns Solution that can be used to compute the total return, or the optimal policy.
 */
template<class SType, class ResponseType = PolicyDeterministic>
inline auto vi_gs(const GRMDP<SType>& mdp, prec_t discount,
                        numvec valuefunction=numvec(0), const ResponseType& response = PolicyDeterministic(),
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
            prec_t newvalue = response.update_solution(solution, states[s], s, solution.valuefunction, discount);

            residual = max(residual, abs(solution.valuefunction[s] - newvalue));
            solution.valuefunction[s] = newvalue;
        }
    }
    solution.residual = residual;
    solution.iterations = i;
    return solution;
}

/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param valuefunction Initial value function
\param response Determines the type of solution method. Allows for customized algorithms
                that can solve various forms of robustness, and risk aversion
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param print_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType, class ResponseType = PolicyDeterministic>
inline auto mpi_jac(const GRMDP<SType>& mdp, prec_t discount,
                const numvec& valuefunction=numvec(0), const ResponseType& response = PolicyDeterministic(),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {

    const auto& states = mdp.get_states();
    typename ResponseType::solution_type solution =
            response.new_solution(states.size(), move(valuefunction));

    // just quit if there are no states
    if( mdp.state_count() == 0) return solution;

    numvec oddvalue = solution.valuefunction;   // set in even iterations (0 is even)
    numvec evenvalue = oddvalue;                // set in odd iterations

    numvec residuals(states.size());

    // residual in the policy iteration part
    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    // use two vectors for value iteration and copy values back and forth
    numvec * sourcevalue = & oddvalue;
    numvec * targetvalue = & evenvalue;

    for(i = 0; i < iterations_pi; i++){

        if(print_progress)
            cout << "Policy iteration " << i << "/" << iterations_pi << ":" << endl;

        swap(targetvalue, sourcevalue);

        prec_t residual_vi = numeric_limits<prec_t>::infinity();

        // update policies
        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            prec_t newvalue = response.update_solution(solution, states[s], s, *sourcevalue, discount);
            residuals[s] = abs((*sourcevalue)[s] - newvalue);
            (*targetvalue)[s] = newvalue;
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
                prec_t newvalue = response.update_value(solution, states[s], s, *sourcevalue, discount);
                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
        if(print_progress) cout << endl << "    Residual (fixed policy): " << residual_vi << endl << endl;
    }
    solution.valuefunction = move(*targetvalue);
    return solution;
}


/// A helper function that simply copies a nature specification across all states
template<class T>
PolicyNature<T> uniform_nature(size_t statecount, NatureResponse<T> nature,
                            T threshold){
    return PolicyNature<T>(vector<NatureInstance<T>>(statecount, make_pair(nature, threshold)));
}

/// A helper function that simply copies a nature specification across all states
template<class Model, class T>
PolicyNature<T> uniform_nature(const Model& m, NatureResponse<T> nature,
                            T threshold){
    return PolicyNature<T>(vector<NatureInstance<T>>(m.state_count(), make_pair(nature, threshold)));
}

}}

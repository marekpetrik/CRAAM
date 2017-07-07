#pragma once

#include "RMDP.hpp"
#include <functional>

#include "cpp11-range-master/range.hpp"

namespace craam {

using namespace std;
using namespace util::lang;


/** 
Function representing the constraints on nature. The function computes
the best response of nature and can be used in value iteration.

This function represents a nature which computes (in general) a randomized
policy (response). If the response is always deterministic, it may be better
to define and use a nature that computes and uses a deterministic response.

The parameters are the q-values v, the reference distribution p, and the threshold.
The function returns the worst-case solution and the objective value. 
*/
typedef vec_scal_t (*NatureResponse)(numvec const& v, numvec const& p, prec_t threshold);


/// L1 robust response
inline vec_scal_t robust_l1(const numvec& v, const numvec& p, prec_t threshold){
    return worstcase_l1(v,p,threshold);
}
/// L1 optimistic response
inline vec_scal_t optimistic_l1(const numvec& v, const numvec& p, prec_t threshold){
    //TODO: this could be faster without copying the vector and just modifying the function
    numvec minusv(v.size());
    transform(begin(v), end(v), begin(minusv), negate<prec_t>());
    auto&& result = worstcase_l1(minusv,p,threshold);
    return make_pair(result.first, -result.second);
}

/// worst outcome, threshold is ignored
inline vec_scal_t robust_unbounded(const numvec& v, const numvec& p, prec_t threshold){
    numvec dist(v.size(),0.0);
    long index = min_element(begin(v), end(v)) - begin(v);
    dist[index] = 1;
    return make_pair(dist,v[index]);
}

/// best outcome, threshold is ignored
inline vec_scal_t optimistic_unbounded(const numvec& v, const numvec& p, prec_t threshold){
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
    return value_action(action, valuefunction, discount, distribution);
}

/**
Computes an ambiguous value (e.g. robust) of the action, depending on the type
of nature that is provided. 

\param action Action for which to compute the value
\param valuefunction State value function to use
\param discount Discount factor
\param nature Method used to compute the response of nature.
\param threshold Threshold parameter for the nature
*/
inline vec_scal_t value_action(const RegularAction& action, const numvec& valuefunction, 
                        prec_t discount, NatureResponse nature, prec_t threshold){

    const numvec& rewards = action.get_outcome().get_rewards();
    const indvec& nonzero_indices = action.get_outcome().get_indices();

    numvec qvalues(rewards.size()); // values for individual states - used by nature.
    #pragma omp simd
    for(auto i : indices(rewards))
        qvalues[i] = rewards[i] + discount * valuefunction[nonzero_indices[i]];

    return nature(qvalues, action.get_outcome().get_probabilities(), threshold);
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
\param threshold Threshold parameter for the nature

\return Outcome distribution and the mean value for the choice of the nature
 */
inline vec_scal_t value_action(const WeightedOutcomeAction& action, numvec const& valuefunction, 
                                prec_t discount, NatureResponse nature, prec_t threshold) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");

    numvec outcomevalues(action.size());
    for(size_t i = 0; i < action.size(); i++){
        outcomevalues[i] = action[i].value(valuefunction, discount);

    return nature(outcomevalues, action.get_distribution(), threshold);
    }
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
\param distribution New distribution to use. Its use depends on the type of 
                    action this function is used with. See value_action for more
                    details. In general, this is a distribution over outcomes or
                    over next states with *nonzero* baseline transition probabilities.

\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class AType>
inline prec_t value_fix_state(const SAState<AType>& state, numvec const& valuefunction, prec_t discount,
                              long actionid, numvec distribution) {
   // this is the terminal state, return 0
    if(state.is_terminal()) return 0;
    if(actionid < 0 || actionid >= (long) state.size()) throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );
    
    const auto& action = state[actionid];
    // cannot assume that the action is valid
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");

    return value_action(action, valuefunction, discount, distribution);
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
\param threshold Threshold parameter for the nature

\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<typename AType>
inline ind_vec_scal_t value_max_state(const SAState<AType>& state, const numvec& valuefunction, 
                                        prec_t discount, NatureResponse nature, prec_t threshold) {
 
    if(state.is_terminal())
        return make_tuple(-1,numvec(),0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();

    long result = -1;
    numvec result_outcome;

    for(size_t i = 0; i < state.get_actions().size(); i++){
        const auto& action = state.get_actions()[i];

        // skip invalid state.get_actions()
        if(!action.is_valid()) continue;

        auto value = value_action(action, valuefunction, discount, nature, threshold);
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
class Solution {
public:
    numvec valuefunction;
    /// index of the actions for each states
    indvec policy;                         
    prec_t residual;
    long iterations;

    Solution(): valuefunction(0), policy(0), residual(-1),iterations(-1) {};

    Solution(const numvec& valuefunction, const indvec& policy, prec_t residual = -1, long iterations = -1) :
        valuefunction(valuefunction), policy(policy), residual(residual), iterations(iterations) {};

    /**
    Computes the total return of the solution given the initial
    distribution.
    \param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const{
        if(initial.max_index() >= (long) valuefunction.size())
            throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};
/** A solution to a robust MDP.  */
class RSolution {
public:
    numvec valuefunction;
    /// index of the action to take for each state
    indvec policy;                         
    /// index of the outcome for each state and action prescribed by the policy
    vector<numvec> outcomes;                      
    prec_t residual;
    long iterations;

    RSolution(): valuefunction(0), policy(0), outcomes(0), residual(-1),iterations(-1) {};

    RSolution(numvec const& valuefunction, const indvec& policy,
             const vector<numvec>& outcomes, prec_t residual = -1, long iterations = -1) :
        valuefunction(valuefunction), policy(policy), outcomes(outcomes), residual(residual),iterations(iterations) {};

    /**
    Computes the total return of the solution given the initial
    distribution.
    \param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const{
        if(initial.max_index() >= (long) valuefunction.size())
            throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.value(valuefunction);
    };
};

/**
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the last state to the first one, the states should be ordered
in the temporal order.

\param discount Discount factor.
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use 
                    all zeros when not provided. Ignored when size is 0.
\param policy Optional. Ignored when size is 0. Possibly a partial policy. Actions are optimized only in 
                 states in which policy = -1, otherwise a fixed value is used. 
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.

\returns Solution that can be used to compute the total return, or the optimal policy.
 */
template<class SType>
inline Solution vi_gs(const GRMDP<SType>& mdp, prec_t discount,
              numvec valuefunction=numvec(0), const indvec& policy=indvec(0),
              unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) {

    const auto& states = mdp.get_states();

    // just quit if there are no states
    if( mdp.state_count() == 0) return Solution();
    // check the dimensions of the policy
    if(!policy.empty() && policy.size() != states.size()) throw invalid_argument("Incorrect dimensions of policy function.");
    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if(!valuefunction.empty()){
        if(valuefunction.size() != states.size()) throw invalid_argument("Incorrect dimensions of value function.");
    }else{
        valuefunction.assign(mdp.state_count(), 0.0);
    }
    indvec newpolicy(states.size());

    // copy the provided policy
    if(!policy.empty())
        copy(begin(policy), end(policy), begin(newpolicy));
    
    // initialize values
    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;   // iterations defined outside to make them reportable

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            const auto& state = states[s];
            prec_t newvalue;
            
            // check whether this state should only be evaluated
            if(policy.empty() || policy[s] < 0){    // optimizing
                tie(newpolicy[s], newvalue) = value_max_state(state, valuefunction,discount);
            }else{ // evaluating (action copied earlier, no need to copy it again)
                newvalue = value_fix_state(state, valuefunction, discount, policy[s]);
            }

            residual = max(residual, abs(valuefunction[s] - newvalue));
            valuefunction[s] = newvalue;
        }
    }
    return Solution(valuefunction,newpolicy,residual,i);
}

/**
*Robust* (or ambiguous) Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the last state to the first one, the states should be ordered
in the temporal order.

\param type Type of realization of the uncertainty
\param discount Discount factor.
\param nature Method used to compute the response of nature.
\param threshold Threshold parameter for the nature
\param valuefunction Initial value function. Passed by value, because it is modified. Optional, use 
                    all zeros when not provided. Ignored when size is 0.
\param policy Optional. Ignored when size is 0. Possibly a partial policy. Actions are optimized only in 
                 states in which policy = -1, otherwise the provided value is used. 
\param natpol Optional. Ignored when size is 0. Possibly a partial policy of nature. Nature is optimized only in 
                 states in which size(natpol[s]) == 0, otherwise the provided value is used. If nature's response
                 is fixed, then also the policy needs to be fixed for that state.
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.

\returns Solution that can be used to compute the total return, or the optimal policy.
 */

template<class SType>
inline RSolution vi_gs(const GRMDP<SType>& mdp, prec_t discount, NatureResponse nature, prec_t threshold,
              numvec valuefunction=numvec(0), const indvec& policy=indvec(0), const vector<numvec>& natpol = vector<numvec>(0),
              unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) {

    const auto& states = mdp.get_states();

  // just quit if there are no states
    if( mdp.state_count() == 0) return RSolution();
    // check the dimensions of the policy
    if(!policy.empty() && policy.size() != states.size()) throw invalid_argument("Incorrect dimensions of policy function.");
    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if(!valuefunction.empty()){
        if(valuefunction.size() != states.size()) throw invalid_argument("Incorrect dimensions of value function.");
    }else{
        valuefunction.assign(mdp.state_count(), 0.0);
    }
    indvec newpolicy(states.size());
    vector<numvec> newnatpol(states.size());

    // copy the provided policy
    if(!policy.empty()) copy(begin(policy), end(policy), begin(newpolicy));
    
    // copy the provided policy for nature
    if(!natpol.empty()) copy(begin(natpol), end(natpol), begin(newnatpol));
    
    // initialize values
    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;   // iterations defined outside to make them reportable

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            const auto& state = states[s];
            prec_t newvalue;
            
            // check whether this state should only be evaluated
            if(policy.empty() || policy[s] < 0){    // optimizing
                assert(natpol[s].empty());   // cannot have a policy for nature if the policy is not provided for the state
                tie(newpolicy[s], newnatpol[s], newvalue) = value_max_state(state, valuefunction, discount, nature, threshold);  
            }else{ // evaluating (action copied earlier, no need to copy it again)
                newvalue = value_fix_state(state, valuefunction, discount, policy[s], natpol[s]);
            }

            residual = max(residual, abs(valuefunction[s] - newvalue));
            valuefunction[s] = newvalue;
        }
    }
    return RSolution(valuefunction,policy,newnatpol,residual,i);
}


/**
Regular modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param valuefunction Initial value function
\param policy Optional. Ignored when size is 0. Possibly a partial policy. Actions are optimized only in 
                 states in which policy = -1, otherwise the provided value is used. 
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param print_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType>
inline Solution mpi_jac(const GRMDP<SType>& mdp, prec_t discount, 
                const numvec& valuefunction=numvec(0),const indvec& policy=indvec(0), 
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {

    const vector<SType>& states = mdp.get_states();

    // quit if there are no states
    if( mdp.state_count() == 0) return Solution();

    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if( !valuefunction.empty() && (valuefunction.size() != mdp.state_count()) ) throw invalid_argument("Incorrect size of value function.");


    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    // construct working policies
    indvec newpolicy(states.size());

    // copy the provided policy
    if(!policy.empty()) copy(begin(policy), end(policy), begin(newpolicy));
    
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
            const auto& state = states[s];

            prec_t newvalue;

            // check whether this state should only be evaluated
            if(policy.empty() || policy[s] < 0){    // optimizing
                tie(newpolicy[s],  newvalue) = value_max_state(state, valuefunction, discount);  
            }else{ // evaluating (action copied earlier, no need to copy it again)
                newvalue = value_fix_state(state, valuefunction, discount, policy[s]);
            }

            residuals[s] = abs((*sourcevalue)[s] - newvalue);
            (*targetvalue)[s] = newvalue;
        }

        residual_pi = *max_element(residuals.begin(),residuals.end());

        if(print_progress) cout << "    Bellman residual: " << residual_pi << endl;

        // the residual is sufficiently small
        if(residual_pi <= maxresidual_pi)
            break;

        if(print_progress) cout << "    Value iteration: ";
        // compute values using value iteration

        for(size_t j = 0; j < iterations_vi && residual_vi > maxresidual_vi; j++){
            if(print_progress) cout << ".";

            swap(targetvalue, sourcevalue);

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                prec_t newvalue = 0;
                const auto& state = states[s];

                newvalue = value_fix_state(state, valuefunction, discount, newpolicy[s]);

                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
        if(print_progress) cout << endl << "    Residual (fixed policy): " << residual_vi << endl << endl;
    }
    numvec & valuenew = *targetvalue;
    return Solution(valuenew,policy,residual_pi,i);
}


/**
*Robust* modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param nature Method used to compute the response of nature.
\param threshold Threshold parameter for the nature
\param valuefunction Initial value function
\param policy Optional. Ignored when size is 0. Possibly a partial policy. Actions are optimized only in 
                 states in which policy = -1, otherwise the provided value is used. 
\param natpol Optional. Ignored when size is 0. Possibly a partial policy of nature. Nature is optimized only in 
                 states in which size(natpol[s]) == 0, otherwise the provided value is used. If nature's response
                 is fixed, then also the policy needs to be fixed for that state.
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param print_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType>
inline RSolution mpi_jac(const GRMDP<SType>& mdp, prec_t discount, NatureResponse nature, prec_t threshold,
                const numvec& valuefunction=numvec(0),const indvec& policy=indvec(0), const vector<numvec>& natpol = vector<numvec>(0),
                unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2,
                bool print_progress=false) {

    const vector<SType>& states = mdp.get_states();

    // quit if there are no states
    if( mdp.state_count() == 0) return RSolution();

    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if( !valuefunction.empty() && (valuefunction.size() != mdp.state_count()) ) throw invalid_argument("Incorrect size of value function.");


    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    // construct working policies
    indvec newpolicy(states.size());
    vector<numvec> newnatpol(states.size());

    // copy the provided policy
    if(!policy.empty()) copy(begin(policy), end(policy), begin(newpolicy));
    
    // copy the provided policy for nature
    if(!natpol.empty()) copy(begin(natpol), end(natpol), begin(newnatpol));

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
            const auto& state = states[s];

            prec_t newvalue;

            // check whether this state should only be evaluated
            if(policy.empty() || policy[s] < 0){    // optimizing
                assert(natpol[s].empty());   // cannot have a policy for nature if the policy is not provided for the state
                tie(newpolicy[s], newnatpol[s], newvalue) = value_max_state(state, valuefunction, discount, nature, threshold);  
            }else{ // evaluating (action copied earlier, no need to copy it again)
                newvalue = value_fix_state(state, valuefunction, discount, policy[s], natpol[s]);
            }

            residuals[s] = abs((*sourcevalue)[s] - newvalue);
            (*targetvalue)[s] = newvalue;
        }

        residual_pi = *max_element(residuals.begin(),residuals.end());

        if(print_progress) cout << "    Bellman residual: " << residual_pi << endl;

        // the residual is sufficiently small
        if(residual_pi <= maxresidual_pi)
            break;

        if(print_progress) cout << "    Value iteration: ";
        // compute values using value iteration

        for(size_t j = 0; j < iterations_vi && residual_vi > maxresidual_vi; j++){
            if(print_progress) cout << ".";

            swap(targetvalue, sourcevalue);

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                prec_t newvalue = 0;
                const auto& state = states[s];

                newvalue = value_fix_state(state, valuefunction, discount, newpolicy[s], newnatpol[s]);

                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
        if(print_progress) cout << endl << "    Residual (fixed policy): " << residual_vi << endl << endl;
    }
    numvec & valuenew = *targetvalue;
    return RSolution(valuenew,policy,newnatpol,residual_pi,i);
}



}

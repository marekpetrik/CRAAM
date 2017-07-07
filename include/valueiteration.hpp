#pragma once

#include "definitions.hpp"
#include "Action.hpp"
#include "State.hpp"
#include "RMDP.hpp"

namespace craam {

using namespace std;

// *******************************************************
// RegularAction computation methods
// *******************************************************

/**
Computes the value of the action.
\param valuefunction State value function to use
\param discount Discount factor
\return Action value
*/
inline prec_t value(const RegularAction& action, const numvec& valuefunction, prec_t discount) 
    {return action.get_outcome().compute_value(valuefunction, discount);}

/**
Computes a value of the action: see RegularAction::value. The
purpose of this method is for the general robust MDP setting.
*/
inline prec_t average(const RegularAction& action, const numvec& valuefunction, prec_t discount) 
    {return value(action,valuefunction, discount);}

/**
Computes a value of the action: see RegularAction::value. The
purpose of this method is for the general robust MDP setting.
*/
inline std::pair<RegularAction::OutcomeId, prec_t>
maximal(const RegularAction& action, const numvec& valuefunction, prec_t discount) 
    {return make_pair(0, value(action,valuefunction, discount));}

/**
Computes a value of the action: see RegularAction::value. The
purpose of this method is for the general robust MDP setting.
*/
inline std::pair<RegularAction::OutcomeId, prec_t>
minimal(const RegularAction& action, const numvec& valuefunction, prec_t discount) 
    {return make_pair(0,value(action, valuefunction, discount));}

/**
Computes a value of the action: see RegularAction::value. The
purpose of this method is for the general robust MDP setting.
*/
inline prec_t fixed(const RegularAction& action, const numvec& valuefunction, prec_t discount, RegularAction::OutcomeId index) 
    {return value(action, valuefunction, discount);}


// *******************************************************
// DiscreteOutcomeAction computation methods
// *******************************************************

/**
Computes the maximal outcome for the value function.
\param valuefunction Value function reference
\param discount Discount factor
\return The index and value of the maximal outcome
 */
inline std::pair<typename DiscreteOutcomeAction::OutcomeId,prec_t>
maximal(const DiscreteOutcomeAction& action, numvec const& valuefunction, prec_t discount) {

    if(action.get_outcomes().empty()) throw invalid_argument("Action with no action.get_outcomes().");
    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1;
    for(size_t i = 0; i < action.get_outcomes().size(); i++){
        const auto& outcome = action.get_outcomes()[i];
        auto value = outcome.compute_value(valuefunction, discount);
        if(value > maxvalue){
            maxvalue = value;
            result = i;
        }
    }
    return make_pair(result,maxvalue);
}

/**
Computes the minimal outcome for the value function
\param valuefunction Value function reference
\param discount Discount factor
\return The index and value of the maximal outcome
*/
inline pair<DiscreteOutcomeAction::OutcomeId,prec_t>
minimal(const DiscreteOutcomeAction& action, numvec const& valuefunction, prec_t discount) {
    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");

    prec_t minvalue = numeric_limits<prec_t>::infinity();
    long result = -1;

    for(size_t i = 0; i < action.get_outcomes().size(); i++){
        const auto& outcome = action.get_outcomes()[i];

        auto value = outcome.compute_value(valuefunction, discount);
        if(value < minvalue){
            minvalue = value;
            result = i;
        }
    }
    return make_pair(result,minvalue);
}

/**
Computes the average outcome using a uniform distribution.
\param valuefunction Updated value function
\param discount Discount factor
\return Mean value of the action
 */
inline prec_t average(const DiscreteOutcomeAction& action, numvec const& valuefunction, prec_t discount) {
    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");

    prec_t averagevalue = 0.0;
    const prec_t weight = 1.0 / prec_t(action.get_outcomes().size());
    for(size_t i = 0; i < action.get_outcomes().size(); ++i)
        averagevalue += weight * action.get_outcomes()[i].compute_value(valuefunction, discount);

    return averagevalue;
}

/**
Computes the action value for a fixed index outcome.
\param valuefunction Updated value function
\param discount Discount factor
\param index Index of the outcome used
\return Value of the action
 */
inline prec_t fixed(const DiscreteOutcomeAction& action, numvec const& valuefunction, prec_t discount,
                   DiscreteOutcomeAction::OutcomeId index) {
    assert(index >= 0l && index < (long) action.get_outcomes().size());
    return action.get_outcomes()[index].compute_value(valuefunction, discount); 
}



// *******************************************************
// WeightedOutcomeAction computation methods
// *******************************************************


/**
Computes the maximal outcome distribution constraints on the nature's distribution.
Template argument nature represents the function used to select the constrained distribution
over the action.get_outcomes().
Does not work when the number of outcomes is zero.
\param valuefunction Value function reference
\param discount Discount factor
\return Outcome distribution and the mean value for the maximal bounded solution
 */
template<NatureConstr nature>
inline pair<WeightedOutcomeAction::OutcomeId,prec_t> maximal(const WeightedOutcomeAction& action, numvec const& valuefunction, prec_t discount) {
    assert(action.get_distribution().size() == action.get_outcomes().size());
    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no action.get_outcomes().");
    numvec outcomevalues(action.get_outcomes().size());
    for(size_t i = 0; i < action.get_outcomes().size(); i++){
        const auto& outcome = action.get_outcomes()[i];
        outcomevalues[i] = - outcome.compute_value(valuefunction, discount);
    }
    auto result = nature(outcomevalues, action.get_distribution(), action.get_threshold());
    result.second = -result.second;
    return result;
}

/**
Computes the minimal outcome distribution constraints on the nature's distribution
Template argument nature represents the function used to select the constrained distribution
over the action.get_outcomes().
Does not work when the number of outcomes is zero.
\param valuefunction Value function reference
\param discount Discount factor
\return Outcome distribution and the mean value for the minimal bounded solution
 */
template<NatureConstr nature>
inline std::pair<typename WeightedOutcomeAction::OutcomeId,prec_t> minimal(const WeightedOutcomeAction& action, numvec const& valuefunction, prec_t discount) {
    assert(action.get_distribution().size() == action.get_outcomes().size());
    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no outcomes");
    numvec outcomevalues(action.get_outcomes().size());
    for(size_t i = 0; i < action.get_outcomes().size(); i++){
        const auto& outcome = action.get_outcomes()[i];
        outcomevalues[i] = outcome.compute_value(valuefunction, discount);
    }
    return nature(outcomevalues, action.get_distribution(), action.get_threshold());
}

/**
Computes the average outcome using a uniform distribution.
\param valuefunction Updated value function
\param discount Discount factor
\return Mean value of the action
 */
prec_t average(const WeightedOutcomeAction& action, numvec const& valuefunction, prec_t discount) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no outcomes");

    prec_t averagevalue = 0.0;
    for(size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += action.get_distribution()[i] * action.get_outcomes()[i].compute_value(valuefunction, discount);

    return averagevalue;
}

/**
Computes the action value for a fixed index outcome.
\param valuefunction Updated value function
\param discount Discount factor
\param index Index of the outcome used
\return Value of the action
 */
prec_t fixed(const WeightedOutcomeAction& action, numvec const& valuefunction, prec_t discount, 
                typename WeightedOutcomeAction::OutcomeId dist) {
    assert(action.get_distribution().size() == action.get_outcomes().size());

    if(action.get_outcomes().empty())
        throw invalid_argument("Action with no outcomes");
    if(dist.size() != action.get_outcomes().size())
        throw invalid_argument("Distribution size does not match number of outcomes");

    prec_t averagevalue = 0.0;
    for(size_t i = 0; i < action.get_outcomes().size(); i++)
        averagevalue += dist[i] * action.get_outcomes()[i].compute_value(valuefunction, discount);

    return averagevalue;
}

// *******************************************************
// State computation methods
// *******************************************************

/**
Finds the maximal optimistic action.
When there are no actions then the return is assumed to be 0.
\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<typename SType, NatureConstr nature>
inline
tuple<typename SType::ActionId,typename SType::OutcomeId,prec_t> 
max_max(const SType& state, const numvec& valuefunction, prec_t discount) {
 
    if(state.is_terminal())
        return make_tuple(-1,typename SType::OutcomeId(),0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    typename SType::OutcomeId result_outcome;

    for(size_t i = 0; i < state.get_actions().size(); i++){
        const auto& action = state.get_actions()[i];

        // skip invalid state.get_actions()
        if(!action.is_valid()) continue;

        auto value = maximal(action, valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = move(value.first);
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

/**
Finds the maximal pessimistic action
When there are no action then the return is assumed to be 0
\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<class SType,NatureConstr nature>
inline
tuple<typename SType::ActionId,typename SType::OutcomeId,prec_t> 
max_min(const SType& state, const numvec& valuefunction, prec_t discount) {
    if(state.is_terminal())
        return make_tuple(-1,typename SType::OutcomeId(),0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    typename SType::OutcomeId result_outcome;

    for(size_t i = 0; i < state.get_actions().size(); i++){
        const auto& action = state.get_actions()[i];

        // skip invalid state.get_actions()
        if(!action.is_valid()) continue;

        auto value = minimal(action, valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = move(value.first);
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

/**
Finds the action with the maximal average return
When there are no actions then the return is assumed to be 0.
\return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
*/
template<class SType,NatureConstr nature>
inline
pair<typename SType::ActionId,prec_t> 
max_average(const SType& state, const numvec& valuefunction, prec_t discount) {
    if(state.is_terminal())
        return make_pair(-1,0.0);
    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    for(size_t i = 0; i < state.get_actions().size(); i++){
        auto const& action = state.get_actions()[i];
        // skip invalid state.get_actions()
        if(!action.is_valid()) continue;
        auto value = average(action, valuefunction, discount);
        if(value > maxvalue){
            maxvalue = value;
            result = i;
        }
    }
    return make_pair(result, maxvalue);
}

/**
Computes the value of a fixed action
\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class SType, NatureConstr nature>
inline
prec_t fixed_average(const SType& state, numvec const& valuefunction, prec_t discount,
                     typename SType::ActionId actionid) {
    // this is the terminal state, return 0
    if(state.is_terminal())
        return 0;
    if(actionid < 0 || actionid >= (long) state.get_actions().size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );
    const auto& action = state.get_actions()[actionid];
    // cannot assume invalid state.get_actions()
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");
    return average(action, valuefunction, discount);
}

/**
Computes the value of a fixed action.
\return Value of state, 0 if it's terminal regardless of the action index
*/
template<class SType,NatureConstr nature>
inline
prec_t fixed_fixed(const SType& state, numvec const& valuefunction, prec_t discount,
                   typename SType::ActionId actionid, typename SType::OutcomeId outcomeid) {
   // this is the terminal state, return 0
    if(state.is_terminal())
        return 0;
    if(actionid < 0 || actionid >= (long) state.get_actions().size())
            throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(state.get_actions().size()) );
    const auto& action = state.get_actions()[actionid];
    // cannot assume invalid state.get_actions()
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");
    return fixed(action, valuefunction, discount, outcomeid);
}
// *******************************************************
// RMDP computation methods
// *******************************************************


/**
Gauss-Seidel variant of value iteration (not parallelized).

This function is suitable for computing the value function of a finite state MDP. If
the states are ordered correctly, one iteration is enough to compute the optimal value function.
Since the value function is updated from the first state to the last, the states should be ordered
in reverse temporal order.

Because this function updates the array value during the iteration, it may be
difficult to paralelize easily.
\param type Type of realization of the uncertainty
\param discount Discount factor.
\param valuefunction Initial value function. Passed by value, because it is modified.
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.
 */
template<class SType, NatureConstr nature>
inline GSolution<typename SType::ActionId, typename SType::OutcomeId>
vi_gs(const GRMDP<SType>& mdp, Uncertainty type,
              prec_t discount,
              numvec valuefunction=numvec(0),
              unsigned long iterations=MAXITER,
              prec_t maxresidual=SOLPREC) {

    const vector<SType>& states = mdp.get_states();


    // just quit if there are no states
    if( mdp.state_count() == 0)
        return GSolution<typename SType::ActionId, typename SType::OutcomeId>
();

    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if(valuefunction.size() > 0){
        if(valuefunction.size() != states.size())
            throw invalid_argument("Incorrect dimensions of value function.");
    }else
        valuefunction.assign(mdp.state_count(), 0.0);


    typename GRMDP<SType>::ActionPolicy policy(states.size());
    typename GRMDP<SType>::OutcomePolicy outcomes(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            const auto& state = states[s];

            tuple<typename SType::ActionId,typename SType::OutcomeId,prec_t> newvalue;

            switch(type){
            case Uncertainty::Robust:
                newvalue = max_min<SType,nature>(state,valuefunction,discount);
                break;
            case Uncertainty::Optimistic:
                newvalue = max_max<SType,nature>(state,valuefunction,discount);
                break;
            case Uncertainty::Average:
                pair<typename SType::ActionId,prec_t> avgvalue =
                    max_average(state,valuefunction,discount);
                newvalue = make_tuple<SType,nature>(avgvalue.first,typename SType::OutcomeId(),avgvalue.second);
                break;
            }

            residual = max(residual, abs(valuefunction[s] - get<2>(newvalue)));
            valuefunction[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }
    }
    return GSolution<typename SType::ActionId, typename SType::OutcomeId>
(valuefunction,policy,outcomes,residual,i);
}

/**
Jacobi variant of value iteration. This method uses OpenMP to parallelize the computation.
\param type Type of realization of the uncertainty
\param valuefunction Initial value function.
\param discount Discount factor.
\param iterations Maximal number of iterations to run
\param maxresidual Stop when the maximal residual falls below this value.
 */
template<class SType, NatureConstr nature>
inline GSolution<typename SType::ActionId, typename SType::OutcomeId>
vi_jac(const GRMDP<SType>& mdp, Uncertainty type,
               prec_t discount,
               const numvec& valuefunction=numvec(0),
                unsigned long iterations=MAXITER,
                prec_t maxresidual=SOLPREC) {

    const vector<SType>& states = mdp.get_states();
    
    // just quit if there are not states
    if( mdp.state_count() == 0)
        return GSolution<typename SType::ActionId, typename SType::OutcomeId> ();

    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if( (valuefunction.size() > 0) && (valuefunction.size() != states.size()) )
        throw invalid_argument("Incorrect size of value function.");

    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    typename GRMDP<SType>::ActionPolicy policy(states.size());
    typename GRMDP<SType>::OutcomePolicy outcomes(states.size());

    numvec residuals(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        numvec & sourcevalue = i % 2 == 0 ? oddvalue  : evenvalue;
        numvec & targetvalue = i % 2 == 0 ? evenvalue : oddvalue;

        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            const auto& state = states[s];

            tuple<typename SType::ActionId,typename SType::OutcomeId,prec_t> newvalue;

            switch(type){
            case Uncertainty::Robust:
                newvalue = max_min<SType,nature>(state,sourcevalue,discount);
                break;
            case Uncertainty::Optimistic:
                newvalue = max_max<SType,nature>(state,sourcevalue,discount);
                break;
            case Uncertainty::Average:
                pair<typename SType::ActionId,prec_t> avgvalue =
                    max_average<SType,nature>(state,sourcevalue,discount);
                newvalue = make_tuple(avgvalue.first,typename SType::OutcomeId(),avgvalue.second);
                break;
            }

            residuals[s] = abs(sourcevalue[s] - get<2>(newvalue));
            targetvalue[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }
        residual = *max_element(residuals.begin(),residuals.end());
    }
    numvec & valuenew = i % 2 == 0 ? oddvalue : evenvalue;
    return GSolution<typename SType::ActionId, typename SType::OutcomeId>
(valuenew,policy,outcomes,residual,i);

}

/**
Modified policy iteration using Jacobi value iteration in the inner loop.
This method generalizes modified policy iteration to robust MDPs.
In the value iteration step, both the action *and* the outcome are fixed.

Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
\param type Type of realization of the uncertainty
\param discount Discount factor
\param valuefunction Initial value function
\param iterations_pi Maximal number of policy iteration steps
\param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
\param iterations_vi Maximal number of inner loop value iterations
\param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
            This value should be smaller than maxresidual_pi
\param show_progress Whether to report on progress during the computation
\return Computed (approximate) solution
 */
template<class SType, NatureConstr nature>
inline GSolution<typename SType::ActionId, typename SType::OutcomeId>
mpi_jac(const GRMDP<SType>& mdp, Uncertainty type,
                prec_t discount,
                const numvec& valuefunction=numvec(0),
                unsigned long iterations_pi=MAXITER,
                prec_t maxresidual_pi=SOLPREC,
                unsigned long iterations_vi=MAXITER,
                prec_t maxresidual_vi=SOLPREC/2,
                bool show_progress=false) {

    const vector<SType>& states = mdp.get_states();

    // just quit if there are no states
    if( mdp.state_count() == 0)
        return GSolution<typename SType::ActionId, typename SType::OutcomeId>();

    // check if the value function is a correct size, and if it is length 0
    // then creates an appropriate size
    if( (valuefunction.size() > 0) && (valuefunction.size() != mdp.state_count()) )
        throw invalid_argument("Incorrect size of value function.");

    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    typename GRMDP<SType>::ActionPolicy policy(states.size());
    typename GRMDP<SType>::OutcomePolicy outcomes(states.size());

    numvec residuals(states.size());

    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    numvec * sourcevalue = & oddvalue;
    numvec * targetvalue = & evenvalue;

    for(i = 0; i < iterations_pi; i++){

        if(show_progress)
            cout << "Policy iteration " << i << "/" << iterations_pi << ":" << endl;

        std::swap<numvec*>(targetvalue, sourcevalue);

        prec_t residual_vi = numeric_limits<prec_t>::infinity();

        // update policies
        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            const auto& state = states[s];

            tuple<typename SType::ActionId,typename SType::OutcomeId,prec_t> newvalue;

            switch(type){
            case Uncertainty::Robust:
                newvalue = max_min<SType,nature>(state,*sourcevalue,discount);
                break;
            case Uncertainty::Optimistic:
                newvalue = max_max<SType,nature>(state,*sourcevalue,discount);
                break;
            case Uncertainty::Average:
                pair<typename SType::ActionId,prec_t> avgvalue =
                    max_average<SType,nature>(state,*sourcevalue,discount);
                newvalue = make_tuple(avgvalue.first,typename SType::OutcomeId(),avgvalue.second);
                break;
            }

            residuals[s] = abs((*sourcevalue)[s] - get<2>(newvalue));
            (*targetvalue)[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }

        residual_pi = *max_element(residuals.begin(),residuals.end());

        if(show_progress)
            cout << "    Bellman residual: " << residual_pi << endl;

        // the residual is sufficiently small
        if(residual_pi <= maxresidual_pi)
            break;

        if(show_progress)
            cout << "    Value iteration: ";
        // compute values using value iteration
        for(size_t j = 0; j < iterations_vi && residual_vi > maxresidual_vi; j++){
            if(show_progress)
                cout << ".";

            swap(targetvalue, sourcevalue);

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                prec_t newvalue = 0;

                switch(type){
                case Uncertainty::Robust:
                case Uncertainty::Optimistic:
                    newvalue = states[s].fixed_fixed(*sourcevalue,discount,policy[s],outcomes[s]);
                    break;
                case Uncertainty::Average:
                    newvalue = states[s].fixed_average(*sourcevalue,discount,policy[s]);
                    break;
                }

                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
        if(show_progress)
            cout << endl << "    Residual (fixed policy): " << residual_vi << endl << endl;
    }
    numvec & valuenew = *targetvalue;
    return GSolution<typename SType::ActionId, typename SType::OutcomeId>(valuenew,policy,outcomes,residual_pi,i);
}

/**
Value function evaluation using Jacobi iteration for a fixed policy.
and nature.

\param valuefunction Initial value function
\param discount Discount factor
\param policy Decision-maker's policy
\param natpolicy Nature's policy
\param iterations Maximal number of inner loop value iterations
\param maxresidual Stop the inner policy iteration when
        the residual drops below this threshold.
\return Computed (approximate) solution (value function)
 */
template<class SType, NatureConstr nature>
inline GSolution<typename SType::ActionId, typename SType::OutcomeId> vi_jac_fix(const GRMDP<SType>& mdp, prec_t discount,
                    const typename GRMDP<SType>::ActionPolicy& policy,
                    const typename GRMDP<SType>::OutcomePolicy& natpolicy,
                    const numvec& valuefunction=numvec(0),
                    unsigned long iterations=MAXITER,
                    prec_t maxresidual=SOLPREC) {

    const vector<SType>& states = mdp.get_states();

    // just quit if there are not states
    if( mdp.state_count() == 0)
        return GSolution<typename SType::ActionId, typename SType::OutcomeId>();

    if(policy.size() != mdp.state_count())
        throw invalid_argument("Dimension of the policy must match the state count.");
    if(natpolicy.size() != mdp.state_count())
        throw invalid_argument("Dimension of the nature's policy must match the state count.");

    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    numvec residuals(states.size());
    prec_t residual = numeric_limits<prec_t>::infinity();

    size_t j; // defined here to be able to report the number of iterations

    numvec * sourcevalue = & oddvalue;
    numvec * targetvalue = & evenvalue;

    for(j = 0; j < iterations && residual > maxresidual; j++){

        swap(targetvalue, sourcevalue);

        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            auto newvalue = fixed_fixed<SType,nature>(states[s],*sourcevalue,discount,policy[s],natpolicy[s]);

            residuals[s] = abs((*sourcevalue)[s] - newvalue);
            (*targetvalue)[s] = newvalue;
        }
        residual = *max_element(residuals.begin(),residuals.end());
    }

    return GSolution<typename SType::ActionId, typename SType::OutcomeId>(*targetvalue,policy,natpolicy,residual,j);
}


}

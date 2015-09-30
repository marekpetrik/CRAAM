#include <assert.h>
#include <numeric>
#include <limits>
#include <algorithm>
#include <stdexcept>

#include "Action.hpp"

using namespace std;


namespace craam {

template<NatureConstr nature>
pair<vector<prec_t>,prec_t>
Action::maximal_cst(vector<prec_t> const& valuefunction, prec_t discount) const{
    /**
       Computes the maximal outcome distribution constraints on the nature's distribution

       Template argument nature represents the function used to select the constrained distribution
       over the outcomes.

       Does not work when the number of outcomes is zero.

       \param valuefunction Value function reference
       \param discount Discount factor

       \return Outcome distribution and the mean value for the maximal bounded solution
     */
    if(distribution.size() != outcomes.size())
        throw range_error("Outcome distribution has incorrect size.");
    if(outcomes.size() == 0)
        throw range_error("Action with no outcomes not allowed when maximizing.");

    vector<prec_t> outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = - outcome.compute_value(valuefunction, discount);
    }

    auto result = nature(outcomevalues, distribution, threshold);
    result.second = - result.second;

    return result;
}

// explicit instantiation of the template
template pair<vector<prec_t>,prec_t> Action::maximal_cst<worstcase_l1>(vector<prec_t> const& valuefunction, prec_t discount) const;

void Action::set_distribution(vector<prec_t> const& distribution, prec_t threshold){
    /**
       Sets the base distribution over the outcomes for this particular action. This distribution
       is used by some methods to compute a limited worst-case solution.

       \param distribution New distribution of outcomes. Must be either the same
                            dimension as the number of outcomes, or length 0. If the length
                            is 0, it is assumed to be a uniform distribution over states.
       \param threshold Bound on the worst-case distribution on the outcomes.
     */

    if(distribution.size() == 0){
        this->distribution = distribution;
        this->threshold = threshold;
        return;
    }

    if(distribution.size() != outcomes.size()){
        throw invalid_argument("invalid distribution size");
    }
    auto sum = accumulate(distribution.begin(),distribution.end(), 0.0);
    if(sum < 0.99 || sum > 1.001){
        throw invalid_argument("invalid distribution");
    }
    auto minimum = *min_element(distribution.begin(),distribution.end());
    if(minimum < 0){
        throw invalid_argument("distribution must be non-negative");
    }

    this->distribution = distribution;
    this->threshold = threshold;
}

pair<long,prec_t> Action::maximal(vector<prec_t> const& valuefunction, prec_t discount) const {
    /**
        Computes the maximal outcome for the value function.

        \param valuefunction Value function reference
        \param discount Discount factor

        \return The index and value of the maximal outcome
     */

    if(outcomes.size() == 0){
        throw range_error("action with no outcomes");
    }

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1;

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];

        auto value = outcome.compute_value(valuefunction, discount);
        if(value > maxvalue){
            maxvalue = value;
            result = i;
        }
    }
    return make_pair(result,maxvalue);
}


pair<long,prec_t> Action::minimal(vector<prec_t> const& valuefunction, prec_t discount) const {
    /**
        Computes the minimal outcome for the value function

        \param valuefunction Value function reference
        \param discount Discount factor

        \return The index and value of the maximal outcome
     */

    if(outcomes.size() == 0){
        throw range_error("action with no outcomes");
    }

    if(outcomes.size() == 0){
        return make_pair(-1,-numeric_limits<prec_t>::infinity());
    }

    prec_t minvalue = numeric_limits<prec_t>::infinity();
    long result = -1;

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];

        auto value = outcome.compute_value(valuefunction, discount);
        if(value < minvalue){
            minvalue = value;
            result = i;
        }
    }
    return make_pair(result,minvalue);
}

prec_t Action::average(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution) const {
    /**
        Computes the minimal outcome for the value function.

        Uses state weights to compute the average. If there is no distribution set, it assumes
        a uniform distribution.

        \param valuefunction Updated value function
        \param discount Discount factor
        \param distribution Reference distribution for computing the mean

        \return Mean value of the action
     */


    if(distribution.size() > 0 && distribution.size() != outcomes.size()){
        throw range_error("invalid size of distribution");
    }

    if(outcomes.size() == 0){
        throw range_error("action with no outcomes");
    }

    prec_t averagevalue = 0.0;

    if(distribution.size() == 0){
        prec_t weight = 1.0 / static_cast<prec_t>(outcomes.size());
        for(size_t i = 0; i < outcomes.size(); i++){
            const auto& outcome = outcomes[i];
            averagevalue += weight * outcome.compute_value(valuefunction, discount);
        }
    }
    else{
        for(size_t i = 0; i < outcomes.size(); i++){
            const auto& outcome = outcomes[i];

            auto value = outcome.compute_value(valuefunction, discount);
            averagevalue += value * distribution[i];
        }
    }
    return averagevalue;
}

prec_t Action::fixed(vector<prec_t> const& valuefunction, prec_t discount, int index) const{
    /**
        Computes the action value for a fixed index outcome.

        \param valuefunction Updated value function
        \param discount Discount factor
        \param index Index of the outcome that is used

        \return Value of the action
     */
    if(index < 0 || index >= (long) outcomes.size()){
        throw range_error("Index is outside of the array");
    }

    const auto& outcome = outcomes[index];
    return outcome.compute_value(valuefunction, discount);
}

Transition& Action::get_transition(long outcomeid){
    /**
        Returns a transition for the outcome to the action. If the transition does
        not exist then it is created.
     */
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");
    if(outcomeid >= (long) outcomes.size())
        outcomes.resize(outcomeid + 1);

    return outcomes[outcomeid];
}

const Transition& Action::get_transition(long outcomeid) const{
    /**
       Returns the transition. The transition must exist.
     */
    if(outcomeid < 0l || outcomeid >= (long) outcomes.size()){
        throw invalid_argument("invalid outcome number");
    }

    return outcomes[outcomeid];
}

void Action::add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward){
    /**
        Adds and outcome to the action. If the outcome does not exist, it is
        created.
     */
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");

    if(outcomeid >= (long) outcomes.size()){
        outcomes.resize(outcomeid + 1);
    }
    outcomes[outcomeid].add_sample(toid, probability, reward);
}

template<NatureConstr nature> pair<vector<prec_t>,prec_t>
Action::minimal_cst(vector<prec_t> const& valuefunction, prec_t discount) const{
    /**
       Computes the minimal outcome distribution constraints on the nature's distribution

       Template argument nature represents the function used to select the constrained distribution
       over the outcomes.

       Returns -infinity when there are no outcomes.

       \param valuefunction Value function reference
       \param discount Discount factor

       \return Outcome distribution and the mean value for the minimal bounded solution
     */

    if(distribution.size() != outcomes.size())
        throw range_error("Outcome distribution has incorrect size.");
    if(outcomes.size() == 0)
        return make_pair(vector<prec_t>(0), -numeric_limits<prec_t>::infinity());

    vector<prec_t> outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = outcome.compute_value(valuefunction, discount);
    }

    return nature(outcomevalues, distribution, threshold);
}

// explicit instantiation of the template
template pair<vector<prec_t>,prec_t> Action::minimal_cst<worstcase_l1>(vector<prec_t> const& valuefunction, prec_t discount) const;


}

#include <assert.h>
#include <numeric>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cmath>

#include "Action.hpp"

using namespace std;

namespace craam {


Action::Action(): threshold(NAN), distribution(0) {
    /**
    Creates an empty action.
    */
}

Action::Action(bool use_distribution):
        threshold(use_distribution ? 0 : NAN),
        distribution(0) {
    /**
    Creates an empty action.

    \param use_distribution Whether to automatically create and scale
                            a distribution function
    */
}

Action::Action(const vector<Transition>& outcomes, bool use_distribution) :
        outcomes(outcomes),
        threshold(use_distribution ? 0 : NAN),
        distribution(outcomes.size(), !outcomes.empty() ?
                                        1.0 / (prec_t) outcomes.size() : 0.0) {
    /**
    Initializes outcomes to the provided vector

    \param use_distribution Whether to automatically create and scale
                            a uniform distribution function to scale to 1
    */
}

template<NatureConstr nature>
pair<numvec,prec_t>
Action::maximal_cst(numvec const& valuefunction, prec_t discount) const{
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

    numvec outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = - outcome.compute_value(valuefunction, discount);
    }

    auto result = nature(outcomevalues, distribution, threshold);
    result.second = - result.second;

    return result;
}

// explicit instantiation of the template
template pair<numvec,prec_t> Action::maximal_cst<worstcase_l1>(numvec const& valuefunction, prec_t discount) const;

void Action::set_distribution(numvec const& distribution){
    /**
       Sets the base distribution over the outcomes for this particular action. This distribution
       is used by some methods to compute a limited worst-case solution.

       \param distribution New distribution of outcomes. Must be either the same
                            dimension as the number of outcomes, or length 0. If the length
                            is 0, it is assumed to be a uniform distribution over states.
     */

    if(distribution.size() != outcomes.size())
        throw invalid_argument("invalid distribution size");

    auto sum = accumulate(distribution.begin(),distribution.end(), 0.0);
    if(sum < 0.99 || sum > 1.001)
        throw invalid_argument("invalid distribution");

    auto minimum = *min_element(distribution.begin(),distribution.end());
    if(minimum < 0)
        throw invalid_argument("distribution must be non-negative");

    this->distribution = distribution;
}

pair<long,prec_t> Action::maximal(numvec const& valuefunction, prec_t discount) const {
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


pair<long,prec_t> Action::minimal(numvec const& valuefunction, prec_t discount) const {
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

prec_t Action::average(numvec const& valuefunction, prec_t discount, numvec const& distribution) const {
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

prec_t Action::fixed(numvec const& valuefunction, prec_t discount, int index) const{
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
        Returns a transition for the outcome to the action. The transition
        must exist
    */
    if(outcomeid < 0l || outcomeid >= (long) outcomes.size())
        throw invalid_argument("invalid outcome number");

    return outcomes[outcomeid];
}

const Transition& Action::get_transition(long outcomeid) const{
    /**
       Returns the transition. The transition must exist.
     */
    if(outcomeid < 0l || outcomeid >= (long) outcomes.size())
        throw invalid_argument("invalid outcome number");

    return outcomes[outcomeid];
}

Transition& Action::create_outcome(long outcomeid){
    /**
    Adds a sufficient number of empty outcomes for the outcomeid
    to be a valid identifier.

    If a distribution is initialized, then it is resized appropriately
    and the weights for new elements are set to 0.
    */

    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");

    if(outcomeid >= (long) outcomes.size())
        outcomes.resize(outcomeid + 1);

    if(!std::isnan(threshold)){
        // got to resize the distribution too
        distribution.resize(outcomeid + 1, 0.0);
    }

    return outcomes[outcomeid];
}

void Action::add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward){
    /**
        Adds and outcome to the action. If the outcome does not exist, it is
        created. Empty transitions are created for all outcome ids
        that are smaller than the new one and do not already exist.

        If a distribution is initialized, then it is resized appropriately
        and the weights for new elements are set to 0.
     */
    create_outcome(outcomeid).add_sample(toid, probability, reward);
}

template<NatureConstr nature> pair<numvec,prec_t>
Action::minimal_cst(numvec const& valuefunction, prec_t discount) const{
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
        return make_pair(numvec(0), -numeric_limits<prec_t>::infinity());

    numvec outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = outcome.compute_value(valuefunction, discount);
    }

    return nature(outcomevalues, distribution, threshold);
}

// explicit instantiation of the template
template pair<numvec,prec_t> Action::minimal_cst<worstcase_l1>(numvec const& valuefunction, prec_t discount) const;

void Action::set_distribution(long outcomeid, prec_t weight){
    /**
        Sets the weight associated with an outcome.
     */
     if(std::isnan(threshold)){
        throw invalid_argument("distribution is not initialized");
     }
     distribution[outcomeid] = weight;

}

void Action::init_distribution(){
    /**
    Sets an initial uniform value for the threshold (0) and the distribution.
    If the distribution already exists, then it is overwritten.
    */

    distribution.clear();
    if(outcomes.size() > 0){
        distribution.resize(outcomes.size(), 1.0/ (prec_t) outcomes.size());
    }
    threshold = 0.0;
}


void Action::normalize_distribution(){
    /**
       Normalizes outcome weights to sum to one. Assumes that the distribution
       is initialized.
     */

    if(std::isnan(threshold)){
        throw invalid_argument("distribution is not initialized.");
    }

    auto weightsum = accumulate(distribution.begin(), distribution.end(), 0.0);

    if(weightsum > 0.0){
        for(auto& p : distribution)
            p /= weightsum;
    }
}

}

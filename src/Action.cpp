#include <numeric>
#include <algorithm>
#include <assert.h>
#include <limits>

#include "Action.hpp"

using namespace std;


namespace craam {

pair<vector<prec_t>,prec_t> worstcase_l1(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t){
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

    if(z.size() <= 0){
        throw invalid_argument("empty arguments");
    }
    if(t < 0.0 || t > 2.0){
        throw invalid_argument("incorrect threshold");
    }
    if(z.size() != q.size()){
        throw invalid_argument("parameter dimensions do not match");
    }


    size_t sz = z.size();

    vector<size_t> smallest = sort_indexes<prec_t>(z);
    vector<prec_t> o(q);

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

    return make_pair(o,r);
}

void Action::set_distribution(vector<prec_t> const& distribution, prec_t threshold){
    /**
     * \brief Sets the base distribution over the outcomes for this particular action. This distribution
     * is used by some methods to compute a limited worst-case solution.
     *
     * \param distribution New distribution of outcomes. Must be either the same dimension as the number
     * of outcomes, or length 0. If the length is 0, it is assumed to be a uniform distribution over states.
     * \param threshold Bound on the worst-case distribution on the outcomes.
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
     * \brief Computes the maximal outcome for the value function
     *
     * \param valuefunction Value function reference
     * \param discount Discount factor
     *
     * \return The index and value of the maximal outcome
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
     * \brief Computes the minimal outcome for the value function
     *
     * \param valuefunction Value function reference
     * \param discount Discount factor
     *
     * \return The index and value of the maximal outcome
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
    /** \brief Computes the minimal outcome for the value function
     *
     * Uses state weights to compute the average. If there is no distribution set, it assumes
     * a uniform distribution.
     *
     * \param valuefunction Updated value function
     * \param discount Discount factor
     * \param distribution Reference distribution for computing the mean
     *
     * \return Mean value of the action
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
     * \brief Computes the action value for a fixed index outcome
     *
     * \param valuefunction Updated value function
     * \param discount Discount factor
     * \param index Index of the outcome that is used
     *
     * \return Value of the action
     */
    if(index < 0 || index >= (long) outcomes.size()){
        throw range_error("index is outside of the array");
    }

    const auto& outcome = outcomes[index];
    return outcome.compute_value(valuefunction, discount);
}

pair<vector<prec_t>,prec_t> Action::maximal_l1(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution, prec_t t) const{
    /**
     * \brief Computes the maximal outcome distribution given l1 constraints on the the distribution
     *
     * \param valuefunction Value function reference
     * \param discount Discount factor
     *
     * \return Outcome distribution and the mean value for the maximal bounded solution
     */

    if(distribution.size() != outcomes.size()){
        throw range_error("distribution not set properly");
    }

    if(outcomes.size() == 0){
        throw range_error("action with no outcomes");
    }

    vector<prec_t> outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = - outcome.compute_value(valuefunction, discount);
    }

    auto result = worstcase_l1(outcomevalues, distribution, t);
    result.second = - result.second;

    return result;
}

pair<vector<prec_t>,prec_t> Action::minimal_l1(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution, prec_t t) const{
    /** \brief Computes the minimal outcome distribution given l1 constraints on the the distribution
     *
     * \param valuefunction Value function reference
     * \param discount Discount factor
     * \param distribution Reference distribution for the L1 bound
     * \param threshold Bound on the L1 deviation
     *
     * \return Outcome distribution and the mean value for the minimal bounded solution
     */

    if(distribution.size() != outcomes.size()){
        throw range_error("distribution not set properly");
    }

    if(outcomes.size() == 0){
        return make_pair(vector<prec_t>(0), -numeric_limits<prec_t>::infinity());
    }

    vector<prec_t> outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = outcome.compute_value(valuefunction, discount);
    }

    return worstcase_l1(outcomevalues, distribution, t);
}

void Action::add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward){
    if(outcomeid < 0){
        throw new invalid_argument("incorrect outcomeid");
    }
    if(outcomeid >= (long) outcomes.size()){
        outcomes.resize(outcomeid + 1);
    }
    outcomes[outcomeid].add_sample(toid, probability, reward);
}

}

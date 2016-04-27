#include "Action.hpp"

#include<numeric>
#include<limits>
#include<algorithm>
#include<stdexcept>

using namespace std;

namespace craam {

Action::Action(): threshold(0), distribution(0), use_distribution(false) {}

Action::Action(bool use_distribution):
        threshold(0),
        distribution(0),
        use_distribution(use_distribution) {}

Action::Action(const vector<Transition>& outcomes, bool use_distribution) :
        outcomes(outcomes),
        threshold(0),
        distribution(outcomes.size(), !outcomes.empty() ?
                                        1.0 / (prec_t) outcomes.size() : 0.0),
        use_distribution(use_distribution) {}

template<NatureConstr nature>
pair<numvec,prec_t>
Action::maximal_cst(numvec const& valuefunction, prec_t discount) const{

    if(distribution.size() != outcomes.size())
        throw range_error("Outcome distribution has incorrect size.");
    if(outcomes.empty())
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
template
pair<numvec,prec_t>
Action::maximal_cst<worstcase_l1>(numvec const& valuefunction, prec_t discount) const;

void Action::set_distribution(numvec const& distribution){

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

    if(outcomes.empty()){
        throw invalid_argument("action with no outcomes");
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

    if(outcomes.empty()){
        throw invalid_argument("action with no outcomes");
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

    if(outcomes.empty()){
        throw invalid_argument("Action with no outcomes");
    }

    if(distribution.size() > 0 && distribution.size() != outcomes.size()){
        throw invalid_argument("Invalid size of distribution");
    }

    prec_t averagevalue = 0.0;
    if(distribution.empty()){
        const prec_t weight = 1.0 / prec_t(outcomes.size());

        for(size_t i = 0; i < outcomes.size(); ++i)
            averagevalue += weight * outcomes[i].compute_value(valuefunction, discount);
    }
    else{
        for(size_t i = 0; i < outcomes.size(); i++)
            averagevalue += distribution[i] * outcomes[i].compute_value(valuefunction, discount);
    }
    return averagevalue;
}

prec_t Action::fixed(numvec const& valuefunction, prec_t discount, int index) const{
    if(index < 0 || index >= (long) outcomes.size()){
        throw range_error("Index is outside of the array");
    }

    const auto& outcome = outcomes[index];
    return outcome.compute_value(valuefunction, discount);
}

Transition& Action::get_transition(long outcomeid){
    if(outcomeid < 0l || outcomeid >= (long) outcomes.size())
        throw invalid_argument("invalid outcome number");

    return outcomes[outcomeid];
}

const Transition& Action::get_transition(long outcomeid) const{
    if(outcomeid < 0l || outcomeid >= (long) outcomes.size())
        throw invalid_argument("invalid outcome number");

    return outcomes[outcomeid];
}

Transition& Action::create_outcome(long outcomeid){
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");

    if(outcomeid >= (long) outcomes.size())
        outcomes.resize(outcomeid + 1);

    if(use_distribution){
        // got to resize the distribution too
        distribution.resize(outcomeid + 1, 0.0);
    }
    return outcomes[outcomeid];
}

void Action::add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward){
    create_outcome(outcomeid).add_sample(toid, probability, reward);
}

template<NatureConstr nature> pair<numvec,prec_t>
Action::minimal_cst(numvec const& valuefunction, prec_t discount) const{
    assert(distribution.size() == outcomes.size());

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes");

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
     if(!use_distribution){
        throw invalid_argument("Distribution is not initialized.");
     }
     distribution[outcomeid] = weight;
}

void Action::init_distribution(){
    distribution.clear();
    if(outcomes.size() > 0){
        distribution.resize(outcomes.size(), 1.0/ (prec_t) outcomes.size());
    }
    threshold = 0.0;
    use_distribution = true;
}


void Action::normalize_distribution(){

    if(!use_distribution){
        throw invalid_argument("Distribution is not initialized.");
    }

    auto weightsum = accumulate(distribution.begin(), distribution.end(), 0.0);

    if(weightsum > 0.0){
        for(auto& p : distribution)
            p /= weightsum;
    }else{
        throw invalid_argument("Distribution sums to 0 and cannot be normalized.");
    }
}

void Action::normalize(){
    for(Transition& t : outcomes){
        t.normalize();
    }
}

/// **************************************************************************************
///  Outcome Management (a helper class)
/// **************************************************************************************


Transition& OutcomeManagement::create_outcome(long outcomeid){
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");

    if(outcomeid >= (long) outcomes.size())
        outcomes.resize(outcomeid + 1);

    return outcomes[outcomeid];
}

void OutcomeManagement::add_outcome(long outcomeid, const Transition& t){
    create_outcome(outcomeid) = t;
}

void OutcomeManagement::normalize(){
    for(Transition& t : outcomes){
        t.normalize();
    }
}

// **************************************************************************************
//  Discrete Outcome Action
// **************************************************************************************

pair<DiscreteOutcomeAction::OutcomeId,prec_t>
DiscreteOutcomeAction::maximal(const numvec& valuefunction, prec_t discount) const {

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes.");

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

pair<DiscreteOutcomeAction::OutcomeId,prec_t>
DiscreteOutcomeAction::minimal(const numvec& valuefunction, prec_t discount) const {

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes.");

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

prec_t DiscreteOutcomeAction::average(const numvec& valuefunction, prec_t discount) const {
    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes.");

    prec_t averagevalue = 0.0;
    const prec_t weight = 1.0 / prec_t(outcomes.size());
    for(size_t i = 0; i < outcomes.size(); ++i)
        averagevalue += weight * outcomes[i].compute_value(valuefunction, discount);

    return averagevalue;
}



/// **************************************************************************************
///  Weighted Outcome Action
/// **************************************************************************************


template<NatureConstr nature>
auto WeightedOutcomeAction<nature>::maximal(const numvec& valuefunction, prec_t discount) const
            -> pair<OutcomeId,prec_t>{

    assert(distribution.size() == outcomes.size());

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes.");

    numvec outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = - outcome.compute_value(valuefunction, discount);
    }

    auto result = nature(outcomevalues, distribution, threshold);
    result.second = -result.second;

    return result;
}

template<NatureConstr nature>
auto WeightedOutcomeAction<nature>::minimal(const numvec& valuefunction, prec_t discount) const
            -> pair<OutcomeId,prec_t>{

    assert(distribution.size() == outcomes.size());

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes");

    numvec outcomevalues(outcomes.size());

    for(size_t i = 0; i < outcomes.size(); i++){
        const auto& outcome = outcomes[i];
        outcomevalues[i] = outcome.compute_value(valuefunction, discount);
    }
    return nature(outcomevalues, distribution, threshold);
}

template<NatureConstr nature>
prec_t WeightedOutcomeAction<nature>::average(numvec const& valuefunction, prec_t discount) const {

    assert(distribution.size() == outcomes.size());

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes");

    prec_t averagevalue = 0.0;
    for(size_t i = 0; i < outcomes.size(); i++)
        averagevalue += distribution[i] * outcomes[i].compute_value(valuefunction, discount);

    return averagevalue;
}

template<NatureConstr nature>
prec_t WeightedOutcomeAction<nature>::fixed(numvec const& valuefunction, prec_t discount,
                                            OutcomeId dist) const{

    assert(distribution.size() == outcomes.size());

    if(outcomes.empty())
        throw invalid_argument("Action with no outcomes");
    if(dist.size() != outcomes.size())
        throw invalid_argument("Distribution size does not match number of outcomes");

    prec_t averagevalue = 0.0;
    for(size_t i = 0; i < outcomes.size(); i++)
        averagevalue += dist[i] * outcomes[i].compute_value(valuefunction, discount);

    return averagevalue;
}

template<NatureConstr nature>
void WeightedOutcomeAction<nature>::set_distribution(numvec const& distribution){

    if(distribution.size() != outcomes.size())
        throw invalid_argument("invalid distribution size");
    prec_t sum = accumulate(distribution.begin(),distribution.end(), 0.0);
    if(sum < 0.99 || sum > 1.001)
        throw invalid_argument("invalid distribution");
    if((*min_element(distribution.begin(),distribution.end())) < 0)
        throw invalid_argument("distribution must be non-negative");

    this->distribution = distribution;
}

template<NatureConstr nature>
Transition& WeightedOutcomeAction<nature>::create_outcome(long outcomeid){
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");
    if(outcomeid >= (long) outcomes.size())
        outcomes.resize(outcomeid + 1);
    // got to resize the distribution too and assign 0 weights
    distribution.resize(outcomeid + 1, 0.0);
    return outcomes[outcomeid];
}


template<NatureConstr nature>
void WeightedOutcomeAction<nature>::set_distribution(long outcomeid, prec_t weight){
     distribution[outcomeid] = weight;
}

template<NatureConstr nature>
void WeightedOutcomeAction<nature>::uniform_distribution(){
    distribution.clear();
    if(outcomes.size() > 0)
        distribution.resize(outcomes.size(), 1.0/ (prec_t) outcomes.size());
    threshold = 0.0;
}

template<NatureConstr nature>
void WeightedOutcomeAction<nature>::normalize_distribution(){
    auto weightsum = accumulate(distribution.begin(), distribution.end(), 0.0);

    if(weightsum > 0.0){
        for(auto& p : distribution)
            p /= weightsum;
    }else{
        throw invalid_argument("Distribution sums to 0 and cannot be normalized.");
    }
}

template<NatureConstr nature>
prec_t WeightedOutcomeAction<nature>::mean_reward(OutcomeId outcomedist) const{
    assert(outcomedist.size() == outcomes.size());

    prec_t result = 0;
    const auto n = outcomes.size();
    for(size_t i=0; i <= n; i++){
        result += outcomedist[i] * outcomes[i].mean_reward();
    }
    return result;
}

template<NatureConstr nature>
Transition WeightedOutcomeAction<nature>::mean_transition(OutcomeId outcomedist) const{
    assert(outcomedist.size() == outcomes.size());

    Transition result;
    const auto n = outcomes.size();
    for(size_t i=0; i <= n; i++){
        outcomes[i].probabilities_addto(outcomedist[i], result);
    }
    return result;
}


/// **************************************************************************************
///  L1 Outcome Action
/// **************************************************************************************

template class WeightedOutcomeAction<worstcase_l1>;

}

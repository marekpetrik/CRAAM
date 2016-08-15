#include "Action.hpp"

#include<numeric>
#include<limits>
#include<algorithm>
#include<stdexcept>
#include<cmath>

#include "cpp11-range-master/range.hpp"

using namespace util::lang;

namespace craam {

using namespace std;

// **************************************************************************************
//  Regular action
// **************************************************************************************

string RegularAction::to_json(long actionid) const{
    string result{"{"};
    result += "\"actionid\" : ";
    result += std::to_string(actionid);
    result += ",\"valid\" :";
    result += std::to_string(valid);
    result += ",\"transition\" : ";
    result += outcome.to_json(-1);
    result += "}";
    return result;
}

// **************************************************************************************
//  Outcome Management (a helper class)
// **************************************************************************************

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


string DiscreteOutcomeAction::to_json(long actionid) const{
    string result{"{"};
    result += "\"actionid\" : ";
    result += std::to_string(actionid);
    result += ",\"valid\" :";
    result += std::to_string(valid);
    result += ",\"outcomes\" : [";
    for(auto oi : indices(outcomes)){
        const auto& o = outcomes[oi];
        result += o.to_json(oi);
        result += ",";
    }
    if(!outcomes.empty()) result.pop_back(); // remove last comma
    result += "]}";
    return result;
}

// **************************************************************************************
//  Weighted Outcome Action
// **************************************************************************************

template<NatureConstr nature>
Transition& WeightedOutcomeAction<nature>::create_outcome(long outcomeid){
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");
    // 1: compute the weight for the new outcome and old ones

    size_t newsize = outcomeid + 1; // new size of the list of outcomes
    size_t oldsize = outcomes.size(); // current size of the set
    if(newsize <= oldsize){// no need to add anything
        return outcomes[outcomeid];
    }
    // new uniform weight for each element
    prec_t newweight = 1.0/prec_t(outcomeid+1);
    // check if need to scale the existing weights
    if(oldsize > 0){
        auto weightsum = accumulate(distribution.begin(), distribution.end(), 0.0);
        // only scale when the sum is not zero
        if(weightsum > 0){
            prec_t normal = (oldsize * newweight) / weightsum;
            transform(distribution.begin(), distribution.end(),distribution.begin(),
                      [normal](prec_t x){return x * normal;});
        }
    }
    outcomes.resize(newsize);
    // got to resize the distribution too and assign weights that are uniform
    distribution.resize(newsize, newweight);
    return outcomes[outcomeid];
}

template<NatureConstr nature>
Transition& WeightedOutcomeAction<nature>::create_outcome(long outcomeid, prec_t weight){
    if(outcomeid < 0)
        throw invalid_argument("Outcomeid must be non-negative.");
    assert(weight >= 0 && weight <= 1);
    
    if(outcomeid >= outcomes.size()){ // needs to resize arrays
        outcomes.resize(outcomeid+1);
        distribution.resize(outcomeid+1);
    }
    set_distribution(outcomeid, weight);
    return outcomes[outcomeid];
}


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
        throw invalid_argument("Invalid distribution size.");
    prec_t sum = accumulate(distribution.begin(),distribution.end(), 0.0);
    if(sum < 0.99 || sum > 1.001)
        throw invalid_argument("Distribution does not sum to 1.");
    if((*min_element(distribution.begin(),distribution.end())) < 0)
        throw invalid_argument("Distribution must be non-negative.");

    this->distribution = distribution;
}


template<NatureConstr nature>
void WeightedOutcomeAction<nature>::set_distribution(long outcomeid, prec_t weight){
     assert(outcomeid >= 0 && (size_t) outcomeid < outcomes.size());
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
bool WeightedOutcomeAction<nature>::is_distribution_normalized() const{
    return abs(1.0-accumulate(distribution.begin(), distribution.end(), 0.0)) < SOLPREC;
}

template<NatureConstr nature>
prec_t WeightedOutcomeAction<nature>::mean_reward(OutcomeId outcomedist) const{
    assert(outcomedist.size() == outcomes.size());

    prec_t result = 0;

    for(size_t i = 0; i < outcomes.size(); i++){
        result += outcomedist[i] * outcomes[i].mean_reward();
    }
    return result;
}

template<NatureConstr nature>
Transition WeightedOutcomeAction<nature>::mean_transition(OutcomeId outcomedist) const{
    assert(outcomedist.size() == outcomes.size());

    Transition result;

    for(size_t i = 0; i < outcomes.size(); i++){
        outcomes[i].probabilities_addto(outcomedist[i], result);
    }
    return result;
}

template<NatureConstr nature>
string WeightedOutcomeAction<nature>::to_json(long actionid) const{
    string result{"{"};
    result += "\"actionid\" : ";
    result += std::to_string(actionid);
    result += ",\"valid\" :";
    result += std::to_string(valid);
    result += ",\"threshold\" : ";
    result += std::to_string(threshold);
    result += ",\"outcomes\" : [";
    for(auto oi : indices(outcomes)){
        const auto& o = outcomes[oi];
        result +=o.to_json(oi);
        result +=",";
    }
    if(!outcomes.empty()) result.pop_back(); // remove last comma
    result += "],\"distribution\" : [";
    for(auto d : distribution){
        result += std::to_string(d);
        result += ",";
    }
    if(!distribution.empty()) result.pop_back(); // remove last comma
    result += "]}";
    return result;
}


// **************************************************************************************
//  L1 Outcome Action
// **************************************************************************************

template class WeightedOutcomeAction<worstcase_l1>;

}

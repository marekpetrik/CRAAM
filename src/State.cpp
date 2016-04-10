#include <limits>
#include <string>

#include "State.hpp"

namespace craam {

tuple<long,long,prec_t> State::max_max(numvec const& valuefunction, prec_t discount) const{

    if(is_terminal())
        return make_tuple(-1,-1,0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    long result_outcome = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        const auto& action = actions[i];
        auto value = action.maximal(valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = value.first;
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

tuple<long,long,prec_t> State::max_min(numvec const& valuefunction, prec_t discount) const{

    if(is_terminal())
        return make_tuple(-1,-1,0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    long result_outcome = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        const auto& action = actions[i];
        auto value = action.minimal(valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = value.first;
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

pair<long,prec_t> State::max_average(numvec const& valuefunction, prec_t discount) const{
    if(is_terminal())
        return make_pair(-1,0.0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        auto const& action = actions[i];
        auto value = action.average(valuefunction, discount);

        if(value > maxvalue){
            maxvalue = value;
            result = i;
        }
    }
    return make_pair(result,maxvalue);
}

// functions used in modified policy iteration
prec_t State::fixed_average(numvec const& valuefunction, prec_t discount, long actionid, numvec const& distribution) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].average(valuefunction, discount, distribution);
}

prec_t State::fixed_average(numvec const& valuefunction, prec_t discount, long actionid) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].average(valuefunction, discount);
}


prec_t State::fixed_fixed(numvec const& valuefunction, prec_t discount, long actionid, long outcomeid) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
            throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].fixed(valuefunction, discount, outcomeid);
}

Transition& State::create_transition(long actionid, long outcomeid){

    if(actionid < 0){
        throw invalid_argument("invalid action id");
    }
    if(actionid >= (long) actions.size()){
        actions.resize(actionid+1);
    }
    return actions[actionid].create_outcome(outcomeid);
}

Transition& State::get_transition(long actionid, long outcomeid){
    /**
       Returns the transition. The transition must exist.
     */
    if(actionid < 0l || actionid >= (long) actions.size()){
        throw invalid_argument("invalid action number");
    }
    return actions[actionid].get_transition(outcomeid);
}

const Transition& State::get_transition(long actionid, long outcomeid) const{
    if(actionid < 0l || actionid >= (long) actions.size()){
        throw invalid_argument("invalid action number");
    }
    return actions[actionid].get_transition(outcomeid);
}

void State::add_action(long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){
    if(actionid < 0){
        throw invalid_argument("invalid action id");
    }
    if(actionid >= (long) actions.size()){
        actions.resize(actionid+1);
    }
    this->actions[actionid].add_outcome(outcomeid, toid, probability, reward);
}

void State::set_thresholds(prec_t threshold){
    for(auto & a : actions){
        a.set_threshold(threshold);
    }
}

template<NatureConstr nature>
tuple<long,numvec,prec_t> State::max_max_cst(numvec const& valuefunction, prec_t discount) const{

    if(is_terminal()){
        return make_tuple(-1,numvec(0),0.0);
    }

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long actionresult = -1l;

    // TODO: change this to an rvalue?
    numvec outcomeresult;

    for(size_t i = 0; i < this->actions.size(); i++){
        const auto& action = actions[i];

        auto outcomevalue = action.maximal_cst<nature>(valuefunction, discount);
        auto value = outcomevalue.second;

        if(value > maxvalue){
            maxvalue = value;
            actionresult = i;
            outcomeresult = outcomevalue.first;
        }
    }
    return make_tuple(actionresult,outcomeresult,maxvalue);
}

template tuple<long,numvec,prec_t>
State::max_max_cst<worstcase_l1>(numvec const& valuefunction, prec_t discount) const;

template<NatureConstr nature>
tuple<long,numvec,prec_t> State::max_min_cst(numvec const& valuefunction, prec_t discount) const{

    if(is_terminal()){
        return make_tuple(-1,numvec(0),0.0);
    }

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long actionresult = -1l;

    numvec outcomeresult;

    for(size_t i = 0; i < this->actions.size(); i++){
        const auto& action = actions[i];

        auto outcomevalue = action.minimal_cst<nature>(valuefunction, discount);
        auto value = outcomevalue.second;

        if(value > maxvalue){
            maxvalue = value;
            actionresult = i;
            outcomeresult = outcomevalue.first;
        }
    }
    return make_tuple(actionresult,outcomeresult,maxvalue);

}

template tuple<long,numvec,prec_t>
State::max_min_cst<worstcase_l1>(numvec const& valuefunction, prec_t discount) const;

void State::normalize(){
    for(Action& a : actions){
        a.normalize();
    }
}

// **************************************************************************************
//  SA State (SA rectangular, also used for a regular MDP)
// **************************************************************************************

template<class AType>
tuple<typename SAState<AType>::ActionId,typename AType::OutcomeId,prec_t>
SAState<AType>::max_max(numvec const& valuefunction, prec_t discount) const{

    if(is_terminal())
        return make_tuple(-1,-1,0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    long result_outcome = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        const auto& action = actions[i];
        auto value = action.maximal(valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = value.first;
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

template<class AType>
tuple<typename SAState<AType>::ActionId,typename AType::OutcomeId,prec_t>
SAState<AType>::max_min(numvec const& valuefunction, prec_t discount) const{

    if(is_terminal())
        return make_tuple(-1,-1,0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    long result_outcome = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        const auto& action = actions[i];
        auto value = action.minimal(valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = value.first;
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

template<class AType>
pair<typename SAState<AType>::ActionId,prec_t>
SAState<AType>::max_average(numvec const& valuefunction, prec_t discount) const{
    if(is_terminal())
        return make_pair(-1,0.0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        auto const& action = actions[i];
        auto value = action.average(valuefunction, discount);

        if(value > maxvalue){
            maxvalue = value;
            result = i;
        }
    }
    return make_pair(result,maxvalue);
}

template<class AType>
prec_t SAState<AType>::fixed_average(numvec const& valuefunction, prec_t discount,
                              typename SAState<AType>::ActionId actionid) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].average(valuefunction, discount);
}

template<class AType>
prec_t SAState<AType>::fixed_fixed(numvec const& valuefunction, prec_t discount,
                            typename SAState<AType>::ActionId actionid,
                            typename AType::OutcomeId outcomeid) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
            throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].fixed(valuefunction, discount, outcomeid);
}

template<class AType>
void SAState<AType>::add_action(long actionid, const AType& action){
    if(actionid < 0){
        throw invalid_argument("invalid action id");
    }
    if(actionid >= (long) actions.size()){
        actions.resize(actionid+1);
    }
    this->actions[actionid] = action;
}

template<class AType>
void SAState<AType>::normalize(){
    for(AType& a : actions){
        a.normalize();
    }
}

template class SAState<RegularAction>;

}

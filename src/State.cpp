#include <limits>
#include <string>

#include "State.hpp"

namespace craam {

// **************************************************************************************
//  SA State (SA rectangular, also used for a regular MDP)
// **************************************************************************************

template<class AType>
auto SAState<AType>::max_max(numvec const& valuefunction, prec_t discount) const
            -> tuple<ActionId,OutcomeId,prec_t> {

    if(is_terminal())
        return make_tuple(-1,OutcomeId(),0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    OutcomeId result_outcome;

    for(size_t i = 0; i < actions.size(); i++){
        const auto& action = actions[i];
        
        // skip invalid actions
        if(!action.is_valid()) continue;
    
        auto value = action.maximal(valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = move(value.first);
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

template<class AType>
auto SAState<AType>::max_min(numvec const& valuefunction, prec_t discount) const
            -> tuple<ActionId,OutcomeId,prec_t> {

    if(is_terminal())
        return make_tuple(-1,OutcomeId(),0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    OutcomeId result_outcome;

    for(size_t i = 0; i < actions.size(); i++){
        const auto& action = actions[i];
        
        // skip invalid actions
        if(!action.is_valid()) continue;

        auto value = action.minimal(valuefunction, discount);
        if(value.second > maxvalue){
            maxvalue = value.second;
            result = i;
            result_outcome = move(value.first);
        }
    }
    return make_tuple(result,result_outcome,maxvalue);
}

template<class AType>
auto SAState<AType>::max_average(numvec const& valuefunction, prec_t discount) const
                -> pair<ActionId,prec_t>{

    if(is_terminal())
        return make_pair(-1,0.0);

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;

    for(size_t i = 0; i < actions.size(); i++){
        auto const& action = actions[i];
        
        // skip invalid actions
        if(!action.is_valid()) continue;

        auto value = action.average(valuefunction, discount);

        if(value > maxvalue){
            maxvalue = value;
            result = i;
        }
    }
    return make_pair(result, maxvalue);
}

template<class AType>
prec_t SAState<AType>::fixed_average(numvec const& valuefunction, prec_t discount,
                              ActionId actionid) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    const auto& action = actions[actionid];

    // cannot assume invalid actions
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");

    return action.average(valuefunction, discount);
}

template<class AType>
prec_t SAState<AType>::fixed_fixed(numvec const& valuefunction, prec_t discount,
                            ActionId actionid, OutcomeId outcomeid) const{

    // this is the terminal state, return 0
    if(is_terminal())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
            throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    const auto& action = actions[actionid];
    // cannot assume invalid actions
    if(!action.is_valid()) throw invalid_argument("Cannot take an invalid action");

    return action.fixed(valuefunction, discount, outcomeid);
}

template<class AType>
AType& SAState<AType>::create_action(long actionid){
    assert(actionid >= 0);

    if(actionid >= (long) actions.size())
        actions.resize(actionid+1);

    return this->actions[actionid];
}

template<class AType>
void SAState<AType>::normalize(){
    for(AType& a : actions)
        a.normalize();
}

template<class AType>
bool SAState<AType>::is_action_outcome_correct(ActionId aid, OutcomeId oid) const{
    if( (aid < 0) || ((size_t)aid >= actions.size()))
        return false;

    return actions[aid].is_outcome_correct(oid);
}


/// **********************************************************************
/// *********************    TEMPLATE DECLARATIONS    ********************
/// **********************************************************************

template class SAState<RegularAction>;
template class SAState<DiscreteOutcomeAction>;
template class SAState<L1OutcomeAction>;

}

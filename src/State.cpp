#include <limits>
#include <string>

#include "State.hpp"

namespace craam {

tuple<long,long,prec_t> State::max_max(numvec const& valuefunction, prec_t discount) const{
    /**
       Finds the maximal optimistic action.

       When there are no action then the return is assumed to be 0.

       \return (Action index, outcome index, value)
     */

    if(this->actions.size() == 0){
        return make_tuple(-1,-1,0);
    }

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    long result_outcome = -1l;

    for(size_t i = 0; i < this->actions.size(); i++){
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
    /**
       Finds the maximal pessimistic action

       When there are no action then the return is assumed to be 0

       \return (Action index, outcome index, value)
     */
    if(this->actions.size() == 0){
        return make_tuple(-1,-1,0);
    }

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long result = -1l;
    long result_outcome = -1l;

    for(size_t i = 0; i < this->actions.size(); i++){
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
    /**
       Finds the action with the maximal average return
     *
       When there are no actions then the return is assumed to be 0.
     *
       \return (Action index, outcome index, value)
     */
    if(actions.size() == 0){
        return make_pair(-1,0.0);
    }

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
    /** Computes the value of a fixed action    */

    // this is the terminal state, return 0
    if(actionid == -1 && 0 == (long) actions.size())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].average(valuefunction, discount, distribution);
}

prec_t State::fixed_average(numvec const& valuefunction, prec_t discount, long actionid) const{
    /** Computes the value of a fixed action    */

    // this is the terminal state, return 0
    if(actionid == -1 && 0 == (long) actions.size())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
        throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].average(valuefunction, discount);
}


prec_t State::fixed_fixed(numvec const& valuefunction, prec_t discount, long actionid, long outcomeid) const{
    /** Computes the value of a fixed action    */

    // this is the terminal state, return 0
    if(actionid == -1 && 0 == (long) actions.size())
        return 0;

    if(actionid < 0 || actionid >= (long) actions.size())
            throw range_error("invalid actionid: " + to_string(actionid) + " for action count: " + to_string(actions.size()) );

    return actions[actionid].fixed(valuefunction, discount, outcomeid);
}

Transition& State::get_transition(long actionid, long outcomeid){
    /**
        Returns the transition; new actions and outcomes are created as necessary.
     */
    if(actionid < 0){
        throw invalid_argument("invalid action id");
    }
    if(actionid >= (long) actions.size()){
        actions.resize(actionid+1);
    }
    return actions[actionid].get_transition(outcomeid);
}

const Transition& State::get_transition(long actionid, long outcomeid) const{
    /**
       Returns the transition. The transition must exist.
     */
    if(actionid < 0l || actionid >= (long) actions.size()){
        throw invalid_argument("invalid action number");
    }
    return actions[actionid].get_transition(outcomeid);
}

void State::add_action(long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){
    /**
       Adds and action (and a whole transition) to the state
     */
    if(actionid < 0){
        throw invalid_argument("invalid action id");
    }
    if(actionid >= (long) actions.size()){
        actions.resize(actionid+1);
    }
    this->actions[actionid].add_outcome(outcomeid, toid, probability, reward);
}

void State::set_thresholds(prec_t threshold){
    /**
       Sets the thresholds for all actions.
     */

    for(auto & a : actions){
        a.set_threshold(threshold);
    }
}

template<NatureConstr nature>
tuple<long,numvec,prec_t> State::max_max_cst(numvec const& valuefunction, prec_t discount) const{
    /**
       Finds the maximal optimistic action given the l1 constraints.

       When there are no action then the return is assumed to be 0.

       \param valuefunction Value function reference
       \param discount Discount factor

       \return Action index, outcome distribution and the mean value for the maximal bounded solution
     */
    if(this->actions.size() == 0){
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
    /**
       Finds the maximal pessimistic action given l1 constraints

       When there are no actions then the return is assumed to be 0

       \param valuefunction Value function reference
       \param discount Discount factor

       \return Outcome distribution and the mean value for the maximal bounded solution

       \return (Action index, outcome index, value)
     */
    if(this->actions.size() == 0){
        return make_tuple(-1,numvec(0),0.0);
    }

    prec_t maxvalue = -numeric_limits<prec_t>::infinity();
    long actionresult = -1l;

    // TODO: change this to an rvalue?
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

}

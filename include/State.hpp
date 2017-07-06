#pragma once

#include "Action.hpp"

#include <utility>
#include <tuple>
#include <vector>
#include <stdexcept>
#include <limits>
#include <string>

#include "cpp11-range-master/range.hpp"


namespace craam {

using namespace std;

// **************************************************************************************
//  SA State (SA rectangular, also used for a regular MDP)
// **************************************************************************************

/**
State for sa-rectangular uncertainty (or no uncertainty) in an MDP

\tparam Type of action used in the state. This type determines the
    type of uncertainty set.
*/
template<class AType>
class SAState{
protected:
    vector<AType> actions;

public:

    /** An identifier for an action for a fixed solution */
    typedef long ActionId;
    /** OutcomeId which comes from outcome*/
    typedef typename AType::OutcomeId OutcomeId;

    SAState() : actions(0) {};
    SAState(const vector<AType>& actions) : actions(actions) {};

    /** Number of actions */
    size_t action_count() const { return actions.size();};

    /** Number of actions */
    size_t size() const { return action_count();};

    /** Creates an action given by actionid if it does not exists.
    Otherwise returns the existing one. */
    AType& create_action(long actionid){
        assert(actionid >= 0);

        if(actionid >= (long) actions.size())
            actions.resize(actionid+1);

        return this->actions[actionid];
    }

    /** Creates an action at the last position of the state */
    AType& create_action() {return create_action(actions.size());};

    /** Returns an existing action */
    const AType& get_action(long actionid) const
                {assert(actionid >= 0 && size_t(actionid) < action_count());
                 return actions[actionid];};

    /** Returns an existing action */
    const AType& operator[](long actionid) const {return get_action(actionid);}

    /** Returns an existing action */
    AType& get_action(long actionid)
                {assert(actionid >= 0 && size_t(actionid) < action_count());
                 return actions[actionid];};

    /** Returns an existing action */
    AType& operator[](long actionid) {return get_action(actionid);}

    /** Returns set of all actions */
    const vector<AType>& get_actions() const {return actions;};

    /** True if the state is considered terminal (no actions). */
    bool is_terminal() const {return actions.empty();};

    /** Normalizes transition probabilities to sum to one. */
    void normalize(){
        for(AType& a : actions)
            a.normalize();
    }

    /** Checks whether the prescribed action and outcome are correct */
    bool is_action_outcome_correct(ActionId aid, OutcomeId oid) const{
        if( (aid < 0) || ((size_t)aid >= actions.size()))
            return false;

        return actions[aid].is_outcome_correct(oid);
    }

    /** Returns the mean reward following the action (and outcome). */
    prec_t mean_reward(ActionId actionid, OutcomeId outcomeid) const{
        return get_action(actionid).mean_reward(outcomeid);
    }

    /** Returns the mean transition probabilities following the action and outcome. */
    Transition mean_transition(ActionId actionid, OutcomeId outcomeid) const{
        return move(get_action(actionid).mean_transition(outcomeid));
    }

    /**
    Finds the maximal optimistic action.
    When there are no action then the return is assumed to be 0.
    \return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
    */
    tuple<ActionId,OutcomeId,prec_t>
    max_max(const numvec& valuefunction, prec_t discount) const{
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

    /**
    Finds the maximal pessimistic action
    When there are no action then the return is assumed to be 0
    \return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
    */
    tuple<ActionId,OutcomeId,prec_t>
    max_min(const numvec& valuefunction, prec_t discount) const{
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

    /**
    Finds the action with the maximal average return
    When there are no actions then the return is assumed to be 0.
    \return (Action index, outcome index, value), 0 if it's terminal regardless of the action index
    */
    pair<ActionId,prec_t>
    max_average(const numvec& valuefunction, prec_t discount) const{
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

    /**
    Computes the value of a fixed action
    \return Value of state, 0 if it's terminal regardless of the action index
    */
    prec_t fixed_average(numvec const& valuefunction, prec_t discount,
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

    /**
    Computes the value of a fixed action.
    \return Value of state, 0 if it's terminal regardless of the action index
    */
    prec_t fixed_fixed(numvec const& valuefunction, prec_t discount,
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

    /** Returns json representation of the state
    \param stateid Includes also state id*/
    string to_json(long stateid = -1) const{
        string result{"{"};
        result += "\"stateid\" : ";
        result += std::to_string(stateid);
        result += ",\"actions\" : [";
        for(auto ai : indices(actions)){
            const auto& a = actions[ai];
            result += a.to_json(ai);
            result += ",";
        }
        if(!actions.empty()) result.pop_back(); // remove last comma
        result += ("]}");
        return result;
    }

};

// **********************************************************************
// *********************    SPECIFIC STATE DEFINITIONS    ***************
// **********************************************************************

/// Regular MDP state with no outcomes
typedef SAState<RegularAction> RegularState;
/// State with uncertain outcomes; unconstrained and now weights
typedef SAState<DiscreteOutcomeAction> DiscreteRobustState;
/// State with uncertain outcomes with L1 constraints on the distribution
typedef SAState<L1OutcomeAction> L1RobustState;
}



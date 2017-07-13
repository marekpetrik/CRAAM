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
    static const bool requires_nature = AType::requires_nature;

    SAState() : actions(0) {};
    SAState(const vector<AType>& actions) : actions(actions) {};

    /** Number of actions */
    size_t action_count() const { return actions.size();};

    /** Number of actions */
    size_t size() const { return action_count();};

    /** 
    Creates an action given by actionid if it does not exists.
    Otherwise returns the existing one. 
    
    All newly created actions are invalid (action.get_valid() = false) and are 
    skipped when computing the state value. Adding transitions to an action
    will make it valid.
    */
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
    bool is_action_correct(long aid, numvec nataction) const{
        if( (aid < 0) || ((size_t)aid >= actions.size()))
            return false;

        return actions[aid].is_nature_correct(nataction);
    }

   /** Checks whether the prescribed action correct */
    bool is_action_correct(long aid) const{
        if( (aid < 0) || ((size_t)aid >= actions.size()))
            return false;
        else
            return true;
    }

    /** Returns the mean reward following the action (and outcome). */
    prec_t mean_reward(long actionid, numvec nataction) const{
        return get_action(actionid).mean_reward(nataction);
    }

    /** Returns the mean reward following the action. */
    prec_t mean_reward(long actionid) const{
        return get_action(actionid).mean_reward();
    }

    /** Returns the mean transition probabilities following the action and outcome. 
    This class assumes a deterministic policy of the decision maker and
    a randomized policy of nature.
    
    \param action Deterministic action of the decision maker 
    \param nataction Randomized action of nature */
    Transition mean_transition(long action, numvec nataction) const{
        return get_action(action).mean_transition(nataction);
    }

    /** Returns the mean transition probabilities following the action and outcome. 
    
    \param action Deterministic action of decision maker */
    Transition mean_transition(long action) const{
        return get_action(action).mean_transition();
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
/// State with uncertain outcomes with L1 constraints on the distribution
typedef SAState<WeightedOutcomeAction> WeightedRobustState;
}


/// helper functions
namespace internal{
    using namespace craam;

    /// checks state and policy with a policy of nature
    template<class SType> 
    bool is_action_correct(const SType& state, long stateid, const std::pair<indvec,vector<numvec>>& policies){
        return state.is_action_correct(policies.first[stateid], policies.second[stateid]);
    }
    
    /// checks state that does not require nature
    template<class SType> 
    bool is_action_correct(const SType& state, long stateid, const indvec& policy){
        static_assert(!SType::requires_nature, "The type of state requires that the policy of nature is also provided.");
        return state.is_action_correct(policy[stateid]);
    }
}

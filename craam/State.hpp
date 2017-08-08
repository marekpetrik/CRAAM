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

Actions within a state are sequentially labeled. That is, adding an action with an id = 3
will also create actions 0,1,2. These additional actions are marked as "invalid" state. This 
means that they will not be used in computations and algorithms will simply skip over them.
Use `create_action` to make and action valid.


\tparam Type of action used in the state. This type determines the
    type of uncertainty set.
*/
template<class AType>
class SAState{
protected:
    /// list of actions
    vector<AType> actions;
    /// whether actions can be used in computation. If false, that means
    /// that they should not be used in algorithms or in computation.
    vector<bool> valid;
    
public:

    SAState() : actions(0), valid(0) {};

    /** Initializes state with actions and sets them all to valid */
    SAState(const vector<AType>& actions) : actions(actions), valid(actions.size(),false) { };

    /** Number of actions */
    size_t action_count() const { return actions.size();};

    /** Number of actions */
    size_t size() const { return action_count();};

    /**
    Creates an action given by actionid if it does not exists.
    Otherwise returns the existing one.

    All newly created actions with id < actionid are created as invalid 
    (action.get_valid() = false). Action actionid is created as valid.
    */
    AType& create_action(long actionid){
        assert(actionid >= 0);
        
        // assumes that the default constructor makes the actions invalid
        if(actionid >= (long) actions.size()){
            actions.resize(actionid+1);
            valid.resize(actionid+1, false);
        }

        // set only the action that is being added as valid
        valid[actionid] = true;
        return actions[actionid];
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

    /// Returns whether the actions is valid
    bool is_valid(long actionid) const {
        assert(actionid < long(valid.size()) && actionid >= 0);
        return valid[actionid];
    };

    /** 
    Set action validity. A valid action can be used in computations. An 
    invalid action is just a placeholder.
    */
    void set_valid(long actionid, bool value = false){
        assert(actionid < long(valid.size()) && actionid >= 0);
        valid[actionid] = value;
    };


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
        if(is_terminal()) return 0;
        else return get_action(actionid).mean_reward(nataction);
    }

    /** Returns the mean reward following the action. */
    prec_t mean_reward(long actionid) const{
        if(is_terminal()) return 0;
        else return get_action(actionid).mean_reward(); 
    }

    /** Returns the mean transition probabilities following the action and outcome.
    This class assumes a deterministic policy of the decision maker and
    a randomized policy of nature.

    \param action Deterministic action of the decision maker
    \param nataction Randomized action of nature */
    Transition mean_transition(long action, numvec nataction) const{
        if(is_terminal()) return Transition();
        else return get_action(action).mean_transition(nataction);
    }

    /** Returns the mean transition probabilities following the action and outcome.

    \param action Deterministic action of decision maker */
    Transition mean_transition(long action) const{
        if(is_terminal()) return Transition();
        else return get_action(action).mean_transition();
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
        return state.is_action_correct(policy[stateid]);
    }
}

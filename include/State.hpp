#pragma once

#include <utility>
#include <tuple>
#include <vector>
#include <stdexcept>

#include "Action.hpp"

using namespace std;

namespace craam {

/** A state in an MDP */
class State {
public:

    State() : actions(0) {};
    State(vector<Action> actions) : actions(actions) {};

    /**
    Finds the maximal optimistic action.
    When there are no action then the return is assumed to be 0.
    \return (Action index, outcome index, value)
    */
    tuple<long,long,prec_t> max_max(numvec const& valuefunction, prec_t discount) const;
    /**
    Finds the maximal pessimistic action
    When there are no action then the return is assumed to be 0
    \return (Action index, outcome index, value)
    */
    tuple<long,long,prec_t> max_min(numvec const& valuefunction, prec_t discount) const;

    /**
    Finds the maximal optimistic action given the l1 constraints.
    When there are no action then the return is assumed to be 0.

    \param valuefunction Value function reference
    \param discount Discount factor
    \return Action index, outcome distribution and the mean value for the maximal bounded solution
    */
    template<NatureConstr nature> tuple<long,numvec,prec_t>
    max_max_cst(numvec const& valuefunction, prec_t discount) const;

    /**
    Finds the maximal pessimistic action given l1 constraints.
    When there are no actions then the return is assumed to be 0

    \param valuefunction Value function reference
    \param discount Discount factor
    \return (Action index, outcome index, value) Outcome distribution
    and the mean value for the maximal bounded solution
    */
    template<NatureConstr nature> tuple<long,numvec,prec_t>
    max_min_cst(numvec const& valuefunction, prec_t discount) const;

    tuple<long,numvec,prec_t> max_max_l1(numvec const& valuefunction, prec_t discount) const{
        return max_max_cst<worstcase_l1>(valuefunction, discount);
    };
    tuple<long,numvec,prec_t> max_min_l1(numvec const& valuefunction, prec_t discount) const{
        return max_min_cst<worstcase_l1>(valuefunction, discount);
    };

    /**
    Finds the action with the maximal average return
    When there are no actions then the return is assumed to be 0.
    \return (Action index, outcome index, value)
    */
    pair<long,prec_t> max_average(numvec const& valuefunction, prec_t discount) const;

    // functions used in modified policy iteration
    /**
    Computes the value of a fixed action
    \return Value of state, 0 if terminal regardless of the action index
    */
    prec_t fixed_average(numvec const& valuefunction, prec_t discount, long actionid, numvec const& distribution) const;
    /**
    Computes the value of a fixed action
    \return Value of state, 0 if terminal regardless of the action index
    */
    prec_t fixed_average(numvec const& valuefunction, prec_t discount, long actionid) const;
    /**
    Computes the value of a fixed action.
    \return Value of state, 0 if terminal regardless of the action index
    */
    prec_t fixed_fixed(numvec const& valuefunction, prec_t discount, long actionid, long outcomeid) const;

    /** Adds and action (and a whole transition) to the state */
    void add_action(long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);

    const Action& get_action(long actionid) const {return actions[actionid];};
    Action& get_action(long actionid) {return actions[actionid];};
    size_t action_count() const { return actions.size();};

    Transition& get_transition(long actionid, long outcomeid);
    /**
    Returns the transition; new actions and outcomes are created
    as necessary.
    */
    Transition& create_transition(long actionid, long outcomeid);

    /** Returns the transition. The transition must exist. */
    const Transition& get_transition(long actionid, long outcomeid) const;

    /** Sets thresholds for all actions.*/
    void set_thresholds(prec_t threshold);

    /** True if the state is considered terminal. That is, it has
    no actions. */
    bool is_terminal() const{
        return actions.empty();
    };

    /** Normalizes transition probabilities to sum to one. */
    void normalize();

public:
    vector<Action> actions;
};


// **************************************************************************************
//  SA State (SA rectangular, also used for a regular MDP)
// **************************************************************************************


}



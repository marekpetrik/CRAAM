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
    State(vector<Action> actions) : actions(actions) { };

    tuple<long,long,prec_t> max_max(numvec const& valuefunction, prec_t discount) const;
    tuple<long,long,prec_t> max_min(numvec const& valuefunction, prec_t discount) const;

    template<NatureConstr nature> tuple<long,numvec,prec_t>
    max_max_cst(numvec const& valuefunction, prec_t discount) const;

    template<NatureConstr nature> tuple<long,numvec,prec_t>
    max_min_cst(numvec const& valuefunction, prec_t discount) const;

    tuple<long,numvec,prec_t> max_max_l1(numvec const& valuefunction, prec_t discount) const{
        return max_max_cst<worstcase_l1>(valuefunction, discount);
    };
    tuple<long,numvec,prec_t> max_min_l1(numvec const& valuefunction, prec_t discount) const{
        return max_min_cst<worstcase_l1>(valuefunction, discount);
    };

    pair<long,prec_t> max_average(numvec const& valuefunction, prec_t discount) const;

    // functions used in modified policy iteration
    prec_t fixed_average(numvec const& valuefunction, prec_t discount, long actionid, numvec const& distribution) const;
    prec_t fixed_average(numvec const& valuefunction, prec_t discount, long actionid) const;
    prec_t fixed_fixed(numvec const& valuefunction, prec_t discount, long actionid, long outcomeid) const;

    void add_action(long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);

    const Action& get_action(long actionid) const {return actions[actionid];};
    Action& get_action(long actionid) {return actions[actionid];};
    size_t action_count() const { return actions.size();};

    Transition& get_transition(long actionid, long outcomeid);
    Transition& create_transition(long actionid, long outcomeid);
    const Transition& get_transition(long actionid, long outcomeid) const;

    void set_thresholds(prec_t threshold);

    bool is_terminal() const{
        /** True if the state is considered terminal. That is, it has
        no actions. */
        return actions.empty();
    };

public: // TODO: change to protected
    vector<Action> actions;
};


}



#ifndef STATE_H
#define STATE_H

#include <utility>
#include <tuple>
#include <vector>
#include <stdexcept>

#include "Action.hpp"

using namespace std;

namespace craam {

class State {
public:
    vector<Action> actions;

    State(){};
    State(vector<Action> actions){
        this->actions = actions;
    }

    tuple<long,long,prec_t> max_max(vector<prec_t> const& valuefunction, prec_t discount) const;
    tuple<long,long,prec_t> max_min(vector<prec_t> const& valuefunction, prec_t discount) const;

    template<pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    tuple<long,vector<prec_t>,prec_t> max_max_cst(vector<prec_t> const& valuefunction, prec_t discount) const{
        /**
           Finds the maximal optimistic action given the l1 constraints.

           When there are no action then the return is assumed to be 0.

           \param valuefunction Value function reference
           \param discount Discount factor

           \return Action index, outcome distribution and the mean value for the maximal bounded solution
         */
        if(this->actions.size() == 0){
            return make_tuple(-1,vector<prec_t>(0),0.0);
        }

        prec_t maxvalue = -numeric_limits<prec_t>::infinity();
        long actionresult = -1l;
        vector<prec_t> outcomeresult;


        for(size_t i = 0; i < this->actions.size(); i++){
            const auto& action = actions[i];

            auto outcomevalue = action.maximal_cst<Nature>(valuefunction, discount);
            auto value = outcomevalue.second;

            if(value > maxvalue){
                maxvalue = value;
                actionresult = i;
                outcomeresult = outcomevalue.first;
            }
        }
        return make_tuple(actionresult,outcomeresult,maxvalue);
    };

    template<pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    tuple<long,vector<prec_t>,prec_t> max_min_cst(vector<prec_t> const& valuefunction, prec_t discount) const{
        /**
           Finds the maximal pessimistic action given l1 constraints

           When there are no actions then the return is assumed to be 0

           \param valuefunction Value function reference
           \param discount Discount factor

           \return Outcome distribution and the mean value for the maximal bounded solution

           \return (Action index, outcome index, value)
         */
        if(this->actions.size() == 0){
            return make_tuple(-1,vector<prec_t>(0),0.0);
        }

        prec_t maxvalue = -numeric_limits<prec_t>::infinity();
        long actionresult = -1l;

        // TODO: change this to an rvalue?
        vector<prec_t> outcomeresult;

        for(size_t i = 0; i < this->actions.size(); i++){
            const auto& action = actions[i];

            auto outcomevalue = action.minimal_cst<Nature>(valuefunction, discount);
            auto value = outcomevalue.second;

            if(value > maxvalue){
                maxvalue = value;
                actionresult = i;
                outcomeresult = outcomevalue.first;
            }
        }
        return make_tuple(actionresult,outcomeresult,maxvalue);

    };

    tuple<long,vector<prec_t>,prec_t> max_max_l1(vector<prec_t> const& valuefunction, prec_t discount) const{
        return max_max_cst<worstcase_l1>(valuefunction, discount);
    };
    tuple<long,vector<prec_t>,prec_t> max_min_l1(vector<prec_t> const& valuefunction, prec_t discount) const{
        return max_min_cst<worstcase_l1>(valuefunction, discount);
    };

    pair<long,prec_t> max_average(vector<prec_t> const& valuefunction, prec_t discount) const;

    // functions used in modified policy iteration
    prec_t fixed_average(vector<prec_t> const& valuefunction, prec_t discount, long actionid, vector<prec_t> const& distribution) const;
    prec_t fixed_average(vector<prec_t> const& valuefunction, prec_t discount, long actionid) const;
    prec_t fixed_fixed(vector<prec_t> const& valuefunction, prec_t discount, long actionid, long outcomeid) const;

    void add_action(long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);

    void set_thresholds(prec_t threshold);
};

}

#endif // STATE_H

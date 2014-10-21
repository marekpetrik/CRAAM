#ifndef ACTION_H
#define ACTION_H

#include <utility>
#include <vector>
#include <stdexcept>

#include "definitions.hpp"
#include "Transition.hpp"

using namespace std;

pair<vector<prec_t>,prec_t> worstcase_l1(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t);

class Action {

public:
    vector<Transition> outcomes;
    vector<prec_t> distribution;
    prec_t threshold;

    Action(){};
    Action(vector<Transition> outcomes){
        this->outcomes = outcomes;
        threshold = -1;
    }

    // plain
    pair<long,prec_t> maximal(vector<prec_t> const& valuefunction, prec_t discount) const;
    pair<long,prec_t> minimal(vector<prec_t> const& valuefunction, prec_t discount) const;

    // average
    prec_t average(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution) const;
    prec_t average(vector<prec_t> const& valuefunction, prec_t discount) const{
        return average(valuefunction,discount,distribution);
    }

    // fixed-outcome
    prec_t fixed(vector<prec_t> const& valuefunction, prec_t discount, int index) const;

    // weighted l1
    pair<vector<prec_t>,prec_t> maximal_l1(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution, prec_t t) const;
    pair<vector<prec_t>,prec_t> maximal_l1(vector<prec_t> const& valuefunction, prec_t discount) const{
        return maximal_l1(valuefunction,discount,distribution,this->threshold);
    };

    pair<vector<prec_t>,prec_t> minimal_l1(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution, prec_t t) const;
    pair<vector<prec_t>,prec_t> minimal_l1(vector<prec_t> const& valuefunction, prec_t discount) const{
        return minimal_l1(valuefunction,discount,distribution,this->threshold);
    };

    void add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward);

    void set_distribution(vector<prec_t> const& distribution, prec_t threshold);

    void set_threshold(prec_t threshold){
        this->threshold = threshold;
    }
};

#endif // ACTION_H

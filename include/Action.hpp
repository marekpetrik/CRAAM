#ifndef ACTION_H
#define ACTION_H

#include <utility>
#include <vector>
#include <limits>

#include "definitions.hpp"
#include "Transition.hpp"

using namespace std;

namespace craam {

class Action {

public:
    vector<Transition> outcomes;
    vector<prec_t> distribution;
    prec_t threshold;

    Action(): threshold(0) {};
    Action(vector<Transition> outcomes) : outcomes(outcomes), threshold(0) {};

    // plain solution
    pair<long,prec_t> maximal(vector<prec_t> const& valuefunction, prec_t discount) const;
    pair<long,prec_t> minimal(vector<prec_t> const& valuefunction, prec_t discount) const;

    // average
    prec_t average(vector<prec_t> const& valuefunction, prec_t discount, vector<prec_t> const& distribution) const;
    prec_t average(vector<prec_t> const& valuefunction, prec_t discount) const{
        return average(valuefunction,discount,distribution);
    }

    // fixed-outcome
    prec_t fixed(vector<prec_t> const& valuefunction, prec_t discount, int index) const;

    // **** weighted constrained
    template<pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    pair<vector<prec_t>,prec_t> 
    maximal_cst(vector<prec_t> const& valuefunction, prec_t discount) const;

    pair<vector<prec_t>,prec_t> maximal_l1(vector<prec_t> const& valuefunction, prec_t discount) const{
        /** Computes the maximal outcome distribution given l1 constraints on the distribution

           Assumes that the number of outcomes is non-zero.

           \param valuefunction Value function reference
           \param discount Discount factor

           \return Outcome distribution and the mean value for the minimal bounded solution
         */

        return maximal_cst<worstcase_l1>(valuefunction, discount);
    };


    template<pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    pair<vector<prec_t>,prec_t> 
    minimal_cst(vector<prec_t> const& valuefunction, prec_t discount) const;

    pair<vector<prec_t>,prec_t> minimal_l1(vector<prec_t> const& valuefunction, prec_t discount) const{
       /** Computes the minimal outcome distribution given l1 constraints on the distribution

       Assumes that the number of outcomes is non-zero.

       \param valuefunction Value function reference
       \param discount Discount factor

       \return Outcome distribution and the mean value for the minimal bounded solution
     */
        return minimal_cst<worstcase_l1>(valuefunction, discount);
    }

    void add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward);

    void set_distribution(vector<prec_t> const& distribution, prec_t threshold);

    void set_threshold(prec_t threshold){
        this->threshold = threshold;
    }
};

}
#endif // ACTION_H

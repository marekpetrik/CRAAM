#pragma once

#include <utility>
#include <vector>
#include <limits>

#include "definitions.hpp"
#include "Transition.hpp"

using namespace std;

namespace craam {

/** An action in an MDP. */
class Action {

public:
    vector<Transition> outcomes;

    Action(): threshold(0) {};
    Action(vector<Transition> outcomes) : outcomes(outcomes), threshold(0) {};

    // plain solution
    pair<long,prec_t> maximal(numvec const& valuefunction, prec_t discount) const;
    pair<long,prec_t> minimal(numvec const& valuefunction, prec_t discount) const;

    // average
    prec_t average(numvec const& valuefunction, prec_t discount, numvec const& distribution) const;
    prec_t average(numvec const& valuefunction, prec_t discount) const{
        return average(valuefunction,discount,distribution);
    }

    // fixed-outcome
    prec_t fixed(numvec const& valuefunction, prec_t discount, int index) const;

    // **** weighted constrained
    template<NatureConstr nature> pair<numvec,prec_t>
    maximal_cst(numvec const& valuefunction, prec_t discount) const;

    pair<numvec,prec_t> maximal_l1(numvec const& valuefunction, prec_t discount) const{
        /** Computes the maximal outcome distribution given l1 constraints on the distribution
           Assumes that the number of outcomes is non-zero.

           \param valuefunction Value function reference
           \param discount Discount factor

           \return Outcome distribution and the mean value for the minimal bounded solution
         */
        return maximal_cst<worstcase_l1>(valuefunction, discount);
    };


    template<NatureConstr nature> pair<numvec,prec_t>
    minimal_cst(numvec const& valuefunction, prec_t discount) const;

    pair<numvec,prec_t> minimal_l1(numvec const& valuefunction, prec_t discount) const{
       /** Computes the minimal outcome distribution given l1 constraints on the distribution

       Assumes that the number of outcomes is non-zero.

       \param valuefunction Value function reference
       \param discount Discount factor

       \return Outcome distribution and the mean value for the minimal bounded solution
     */
        return minimal_cst<worstcase_l1>(valuefunction, discount);
    }

    void add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward);
    const Transition& get_outcome(long outcomeid) const {return outcomes[outcomeid];};
    Transition& get_outcome(long outcomeid) {return outcomes[outcomeid];};
    size_t outcome_count() const {return outcomes.size();};

    Transition& get_transition(long outcomeid);
    const Transition& get_transition(long outcomeid) const;
    
    void set_distribution(numvec const& distribution);
    void set_distribution(long outcomeid, prec_t weight);
    const numvec& get_distribution() const {return distribution;};
    void normalize_distribution();

    prec_t get_threshold() const {return threshold;};
    void set_threshold(prec_t threshold){ this->threshold = threshold; }

protected:
    numvec distribution;
    prec_t threshold;
};

}


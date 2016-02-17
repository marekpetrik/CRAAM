#pragma once

#include <utility>
#include <vector>
#include <limits>
#include <cassert>

#include "definitions.hpp"
#include "Transition.hpp"

using namespace std;

namespace craam {

/**
An action in an MDP.

The worst-case behavior is parameterized by a base distribution and a threshold value.
For example, the worst case can be limited by a threshold of L1 norm deviation from
the provided distribution.

It is important to note that the base distribution over actions is not initialized
by default.

The fact that the distribution is not being used is indicated by threshold being NaN.

If the distribution is used, then it is initialized to be uniform over the provided elements;
when a new outcome is added, then its weight in the distribution is 0.
*/
class Action {

protected:
    // TODO: restrict from resizing!
    /** Do not freely modify this value */
    vector<Transition> outcomes;

public:
    /** Creates an empty action. */
    Action();

    /**
    Creates an empty action.
    \param use_distribution Whether to automatically create and scale
                            a distribution function
    */
    Action(bool use_distribution);
    /**
    Initializes outcomes to the provided vector

    \param use_distribution Whether to automatically create and scale
                            a uniform distribution function to scale to 1
    */
    Action(const vector<Transition>& outcomes, bool use_distribution=false);

    // plain solution
    /**
    Computes the maximal outcome for the value function.

    \param valuefunction Value function reference
    \param discount Discount factor

    \return The index and value of the maximal outcome
     */
    pair<long,prec_t> maximal(numvec const& valuefunction, prec_t discount) const;
    /**
    Computes the minimal outcome for the value function
    \param valuefunction Value function reference
    \param discount Discount factor
    \return The index and value of the maximal outcome
    */
    pair<long,prec_t> minimal(numvec const& valuefunction, prec_t discount) const;

    // average
    /**
    Computes the minimal outcome for the value function.

    Uses state weights to compute the average. If there is no distribution set, it assumes
    a uniform distribution.

    \param valuefunction Updated value function
    \param discount Discount factor
    \param distribution Reference distribution for computing the mean

    \return Mean value of the action
     */

    prec_t average(numvec const& valuefunction, prec_t discount, const numvec& distribution) const;
    prec_t average(numvec const& valuefunction, prec_t discount) const{
        return average(valuefunction,discount,distribution);
    }
    // fixed-outcome
    /**
    Computes the action value for a fixed index outcome.

    \param valuefunction Updated value function
    \param discount Discount factor
    \param index Index of the outcome that is used

    \return Value of the action
     */
    prec_t fixed(numvec const& valuefunction, prec_t discount, int index) const;


    // **** weighted constrained
    /**
    Computes the maximal outcome distribution constraints on the nature's distribution

    Template argument nature represents the function used to select the constrained distribution
    over the outcomes.

    Does not work when the number of outcomes is zero.

    \param valuefunction Value function reference
    \param discount Discount factor

    \return Outcome distribution and the mean value for the maximal bounded solution
     */
    template<NatureConstr nature> pair<numvec,prec_t>
    maximal_cst(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the maximal outcome distribution given l1 constraints on the distribution
    Assumes that the number of outcomes is non-zero.

    \param valuefunction Value function reference
    \param discount Discount factor
    \return Outcome distribution and the mean value for the minimal bounded solution
    */
    pair<numvec,prec_t> maximal_l1(numvec const& valuefunction, prec_t discount) const{
        return maximal_cst<worstcase_l1>(valuefunction, discount);
    };

    /**
    Computes the minimal outcome distribution constraints on the nature's distribution

    Template argument nature represents the function used to select the constrained distribution
    over the outcomes.

    Returns -infinity when there are no outcomes.

    \param valuefunction Value function reference
    \param discount Discount factor

    \return Outcome distribution and the mean value for the minimal bounded solution
     */
    template<NatureConstr nature> pair<numvec,prec_t>
    minimal_cst(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the minimal outcome distribution given l1 constraints on the distribution.
    Assumes that the number of outcomes is non-zero.

    \param valuefunction Value function reference
    \param discount Discount factor
    \return Outcome distribution and the mean value for the minimal bounded solution
    */
    pair<numvec,prec_t> minimal_l1(numvec const& valuefunction, prec_t discount) const{
        return minimal_cst<worstcase_l1>(valuefunction, discount);
    }

    /**
    Adds and outcome to the action. If the outcome does not exist, it is
    created. Empty transitions are created for all outcome ids
    that are smaller than the new one and do not already exist.

    If a distribution is initialized, then it is resized appropriately
    and the weights for new elements are set to 0.
    */
    void add_outcome(long outcomeid, long toid, prec_t probability, prec_t reward);
    /**
    Adds a sufficient number of empty outcomes for the outcomeid
    to be a valid identifier.

    If a distribution is initialized, then it is resized appropriately
    and the weights for new elements are set to 0.
    */
    Transition& create_outcome(long outcomeid);

    const Transition& get_outcome(long outcomeid) const {return outcomes[outcomeid];};
    Transition& get_outcome(long outcomeid) {return outcomes[outcomeid];};
    size_t outcome_count() const {return outcomes.size();};

    /** Returns a transition for the outcome to the action. The transition must exist. */
    Transition& get_transition(long outcomeid);
    /** Returns the transition. The transition must exist. */
    const Transition& get_transition(long outcomeid) const;

    /**
    Sets an initial uniform value for the threshold (0) and the distribution.
    If the distribution already exists, then it is overwritten.
    */
    void init_distribution();
    /**
    Sets the base distribution over the outcomes for this particular action. This distribution
    is used by some methods to compute a limited worst-case solution.

    \param distribution New distribution of outcomes. Must be either the same
                        dimension as the number of outcomes, or length 0. If the length
                        is 0, it is assumed to be a uniform distribution over states.
     */
    void set_distribution(const numvec& distribution);
    /**
    Sets the base distribution over the outcomes for this particular action. This distribution
    is used by some methods to compute a limited worst-case solution.

    \param distribution New distribution of outcomes. Must be either the same
                        dimension as the number of outcomes, or length 0. If the length
                        is 0, it is assumed to be a uniform distribution over states.
     */
    void set_distribution(long outcomeid, prec_t weight);
    const numvec& get_distribution() const {return distribution;};
    /**
    Normalizes outcome weights to sum to one. Assumes that the distribution
    is initialized. Exception is thrown if the distribution sums
    to zero.
    */
    void normalize_distribution();

    /** Whether distribution over outcomes is being used */
    bool get_use_distribution() const {return use_distribution;}

    prec_t get_threshold() const {return threshold;};
    void set_threshold(prec_t threshold){ this->threshold = threshold; }

    /** Returns the list of outcomes */
    const vector<Transition>& get_outcomes() const {return outcomes;};

    /** Normalizes transitions for outcomes */
    void normalize();

protected:
    prec_t threshold;
    numvec distribution;
    bool use_distribution;
};

// **************************************************************************************
// Regular action
// **************************************************************************************

/**
 * Action in a regular MDP. There is no uncertainty and
 * the action contains only a single outcome.
 */
class RegularAction{
protected:
    Transition outcome;

public:
    /** Type of an identifier for an outcome. It is ignored for the simple action. */
    typedef int OutcomeId;

    /** Creates an empty action. */
    RegularAction(){};

    /** Initializes outcomes to the provided transition vector */
    RegularAction(const Transition& outcome) : outcome(outcome) {};

    /**
    Computes the value of the action.
    \param valuefunction State value function to use
    \param discount Discount factor
    \return Action value
    */
    prec_t value(const numvec& valuefunction, prec_t discount) const
        {return outcome.compute_value(valuefunction, discount);};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    prec_t average_value(const numvec& valuefunction, prec_t discount) const
        {return value(valuefunction, discount);};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    pair<RegularAction::OutcomeId, prec_t>
    maximal_value(const numvec& valuefunction, prec_t discount) const
        {return make_pair(0,value(valuefunction, discount));};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    pair<RegularAction::OutcomeId, prec_t>
    minimal_value(const numvec& valuefunction, prec_t discount) const
        {return make_pair(0,value(valuefunction, discount));};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    prec_t fixed_value(const numvec& valuefunction, prec_t discount,
                RegularAction::OutcomeId index) const
        {return value(valuefunction, discount);};

    /** Returns a transition for the outcome to the action. The transition must exist. */
    Transition& get_transition() {return outcome;};

    /** Returns the transition. The transition must exist. */
    const Transition& get_transition() const {return outcome;};

    /** Normalizes transition probabilities */
    void normalize() {outcome.normalize();};
};


// **************************************************************************************
//  Outcome Management (a helper class)
// **************************************************************************************

/**
A class that manages creation and access to outcomes to be used by actions.
*/
class OutcomeManagement{

protected:
    /** List of possible outcomes */
    vector<Transition> outcomes;

public:
    /** Empty list of outcomes */
    OutcomeManagement() {};

    /** Initializes with a list of outcomes */
    OutcomeManagement(const vector<Transition>& outcomes) : outcomes(outcomes) {};

    /**
    Adds a sufficient number of empty outcomes for the outcomeid to be a valid identifier.
    This method is virtual to make overloading safer.
    */
    virtual Transition& create_outcome(long outcomeid);

    /** Returns a transition for the outcome to the action. The transition must exist. */
    const Transition& get_outcome(long outcomeid) const {
        assert((outcomeid >= 0l && outcomeid < (long) outcomes.size()));
        return outcomes[outcomeid];};

    /** Returns a transition for the outcome to the action. The transition must exist. */
    Transition& get_outcome(long outcomeid) {
        assert((outcomeid >= 0l && outcomeid < (long) outcomes.size()));
        return outcomes[outcomeid];};
    size_t outcome_count() const {return outcomes.size();};


    /** Adds an outcome defined by the transition.
    \param outcomeid Id of the new outcome. Intermediate ids are created empty
    \param t Transition that defines the outcome*/
    void add_outcome(long outcomeid, const Transition& t);

    /** Adds an outcome defined by the transition as the last outcome.
    \param t Transition that defines the outcome*/
    void add_outcome(const Transition& t);

    /** Returns the list of outcomes */
    const vector<Transition>& get_outcomes() const {return outcomes;};

    /** Normalizes transitions for outcomes */
    void normalize();
};

// **************************************************************************************
//  Discrete Outcome Action
// **************************************************************************************

/**
An action in the robust MDP with discrete outcomes.
*/
class DiscreteOutcomeAction : public OutcomeManagement {

public:
    /** Type of an identifier for an outcome. It is ignored for the simple action. */
    typedef long OutcomeId;

    /** Creates an empty action. */
    DiscreteOutcomeAction() {};

    /**
    Initializes outcomes to the provided vector
    */
    DiscreteOutcomeAction(const vector<Transition>& outcomes)
        : OutcomeManagement(outcomes){};

    /**
    Computes the maximal outcome for the value function.
    \param valuefunction Value function reference
    \param discount Discount factor
    \return The index and value of the maximal outcome
     */
    pair<DiscreteOutcomeAction::OutcomeId,prec_t>
    maximal_value(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the minimal outcome for the value function
    \param valuefunction Value function reference
    \param discount Discount factor
    \return The index and value of the maximal outcome
    */
    pair<DiscreteOutcomeAction::OutcomeId,prec_t>
    minimal_value(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the average outcome using a uniform distribution.
    \param valuefunction Updated value function
    \param discount Discount factor
    \return Mean value of the action
     */
    prec_t average_value(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the action value for a fixed index outcome.
    \param valuefunction Updated value function
    \param discount Discount factor
    \param index Index of the outcome used
    \return Value of the action
     */
    prec_t fixed_value(numvec const& valuefunction, prec_t discount,
                       DiscreteOutcomeAction::OutcomeId index) const{
        assert(index >= 0l && index < (long) outcomes.size());
        return outcomes[index].compute_value(valuefunction, discount); };

};

// **************************************************************************************
//  Weighted Outcome Action
// **************************************************************************************


/**
An action in a robust MDP in which the outcomes are defined by a weighted function
and a threshold. The uncertain behavior is parameterized by a base distribution
and a threshold value. An example may be a worst case computation:
    min { u^T v | || u - d ||_1 <=  t}
where v are the values for individual outcomes, d is the base distribution, and
t is the threshold. See L1Action for an example of an instance of this template class.

The function that determines the uncertainty set is defined by NatureConstr template parameter.

The distribution is initialized to be uniform over the provided elements;
when a new outcome is added, then its weight in the distribution is 0.
*/
template<NatureConstr nature>
class WeightedOutcomeAction : OutcomeManagement{

protected:
    /** Threshold */
    prec_t threshold;
    /** Weights used in computing the worst/best case */
    numvec distribution;

public:
    /** Type of the outcome identification */
    typedef numvec OutcomeId;

    /** Creates an empty action. */
    WeightedOutcomeAction()
        : OutcomeManagement(), threshold(0), distribution(0) {};

    /** Initializes outcomes to the provided vector */
    WeightedOutcomeAction(const vector<Transition>& outcomes)
        : OutcomeManagement(outcomes), threshold(0), distribution(0) {};

    /**
    Computes the maximal outcome distribution constraints on the nature's distribution.
    Template argument nature represents the function used to select the constrained distribution
    over the outcomes.
    Does not work when the number of outcomes is zero.
    \param valuefunction Value function reference
    \param discount Discount factor
    \return Outcome distribution and the mean value for the maximal bounded solution
     */
    pair<typename WeightedOutcomeAction::OutcomeId,prec_t>
    maximal_value(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the minimal outcome distribution constraints on the nature's distribution
    Template argument nature represents the function used to select the constrained distribution
    over the outcomes.
    Does not work when the number of outcomes is zero.
    \param valuefunction Value function reference
    \param discount Discount factor
    \return Outcome distribution and the mean value for the minimal bounded solution
     */
    pair<typename WeightedOutcomeAction<nature>::OutcomeId,prec_t>
    minimal_value(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the average outcome using a uniform distribution.
    \param valuefunction Updated value function
    \param discount Discount factor
    \return Mean value of the action
     */
    prec_t average_value(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the action value for a fixed index outcome.
    \param valuefunction Updated value function
    \param discount Discount factor
    \param index Index of the outcome used
    \return Value of the action
     */
    prec_t fixed_value(numvec const& valuefunction, prec_t discount,
                       typename WeightedOutcomeAction<nature>::OutcomeId dist) const;

    /**
    Adds a sufficient number of empty outcomes for the outcomeid to be a valid identifier.
    This override also handles properly resizing the distribution.
    */
    Transition& create_outcome(long outcomeid) override;

    /**
    Sets the base distribution over the outcomes.
    \param distribution New distribution of outcomes.
     */
    void set_distribution(const numvec& distribution);

    /**
    Sets weight for a particular outcome.
    \param distribution New distribution of outcomes.
    \param weight New weight
     */
    void set_distribution(long outcomeid, prec_t weight);

    /** Returns the baseline distribution. */
    const numvec& get_distribution() const {return distribution;};

    /**
    Normalizes outcome weights to sum to one. Assumes that the distribution
    is initialized. Exception is thrown if the distribution sums
    to zero.
    */
    void normalize_distribution();

    /**
    Sets an initial uniform value for the threshold (0) and the distribution.
    If the distribution already exists, then it is overwritten.
    */
    void uniform_distribution();

    /** Returns threshold value */
    prec_t get_threshold() const {return threshold;};

    /** Sets threshold value */
    void set_threshold(prec_t threshold){ this->threshold = threshold; }
};


}


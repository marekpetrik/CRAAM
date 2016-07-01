#pragma once

#include <utility>
#include <vector>
#include <limits>
#include <cassert>
#include <string>


#include "definitions.hpp"
#include "Transition.hpp"

using namespace std;

namespace craam {

/// **************************************************************************************
/// *** Regular action
/// **************************************************************************************

/**
 * Action in a regular MDP. There is no uncertainty and
 * the action contains only a single outcome.
 */
class RegularAction{
protected:
    Transition outcome;

public:
    /** Type of an identifier for an outcome. It is ignored for the simple action. */
    typedef long OutcomeId;

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
    prec_t average(const numvec& valuefunction, prec_t discount) const
        {return value(valuefunction, discount);};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    pair<OutcomeId, prec_t>
    maximal(const numvec& valuefunction, prec_t discount) const
        {return make_pair(0,value(valuefunction, discount));};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    pair<OutcomeId, prec_t>
    minimal(const numvec& valuefunction, prec_t discount) const
        {return make_pair(0,value(valuefunction, discount));};

    /**
    Computes a value of the action: see RegularAction::value. The
    purpose of this method is for the general robust MDP setting.
    */
    prec_t fixed(const numvec& valuefunction, prec_t discount, OutcomeId index) const
        {return value(valuefunction, discount);};

    /** Returns the outcomes. */
    vector<Transition> get_outcomes() const {return vector<Transition>{outcome};};

    /** Returns the single outcome. */
    const Transition& get_outcome(long outcomeid) const {assert(outcomeid == 0); return outcome;};

    /** Returns the single outcome. */
    const Transition& get_outcome() const {return outcome;};

    /** Returns the single outcome. */
    Transition& get_outcome() {return outcome;};

    /**
    Adds a sufficient number of empty outcomes for the outcomeid to be a valid identifier.
    This method does nothing in this action.
    */
    Transition& create_outcome(long outcomeid){assert(outcomeid == 0);return outcome;};

    /** Returns the single outcome. */
    Transition& get_outcome(long outcomeid) {assert(outcomeid == 0);return outcome;};

    /** Normalizes transition probabilities */
    void normalize() {outcome.normalize();};

    /** Returns number of outcomes (1). */
    size_t outcome_count() const {return 1;};

    /** Appends a string representation to the argument */
    void to_string(string& result) const{
        result.append("1(reg)");
    };

    /** Whether the provided outcome is valid */
    bool is_outcome_correct(OutcomeId oid) const {return oid == 0;};

    /** Returns the mean reward from the transition. */
    prec_t mean_reward(OutcomeId) const { return outcome.mean_reward();};

    /** Returns the mean transition probabilities. Ignore rewards. */
    Transition mean_transition(OutcomeId) const {return outcome;};
};


/// **************************************************************************************
///  Outcome Management (a helper class)
/// **************************************************************************************

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

    /** Returns number of outcomes. */
    size_t outcome_count() const {return outcomes.size();};
    /** Returns number of outcomes. */
    size_t size() const {return outcome_count();};


    /** Adds an outcome defined by the transition.
    \param outcomeid Id of the new outcome. Intermediate ids are created empty
    \param t Transition that defines the outcome*/
    void add_outcome(long outcomeid, const Transition& t);

    /** Adds an outcome defined by the transition as the last outcome.
    \param t Transition that defines the outcome*/
    void add_outcome(const Transition& t){add_outcome(outcomes.size(), t);};


    /** Returns the list of outcomes */
    const vector<Transition>& get_outcomes() const {return outcomes;};

    /** Normalizes transitions for outcomes */
    void normalize();

    /** Appends a string representation to the argument */
    void to_string(string& result) const{
        result.append(std::to_string(get_outcomes().size()));
    }

};

/// **************************************************************************************
///  Discrete Outcome Action
/// **************************************************************************************

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
    maximal(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the minimal outcome for the value function
    \param valuefunction Value function reference
    \param discount Discount factor
    \return The index and value of the maximal outcome
    */
    pair<DiscreteOutcomeAction::OutcomeId,prec_t>
    minimal(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the average outcome using a uniform distribution.
    \param valuefunction Updated value function
    \param discount Discount factor
    \return Mean value of the action
     */
    prec_t average(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the action value for a fixed index outcome.
    \param valuefunction Updated value function
    \param discount Discount factor
    \param index Index of the outcome used
    \return Value of the action
     */
    prec_t fixed(numvec const& valuefunction, prec_t discount,
                       DiscreteOutcomeAction::OutcomeId index) const{
        assert(index >= 0l && index < (long) outcomes.size());
        return outcomes[index].compute_value(valuefunction, discount); };

    /** Whether the provided outcome is valid */
    bool is_outcome_correct(OutcomeId oid) const
        {return (oid >= 0) && ((size_t) oid < outcomes.size());};


    /** Returns the mean reward from the transition. */
    prec_t mean_reward(OutcomeId oid) const { return outcomes[oid].mean_reward();};

    /** Returns the mean transition probabilities */
    Transition mean_transition(OutcomeId oid) const {return outcomes[oid];};
};

/// **************************************************************************************
///  Weighted Outcome Action
/// **************************************************************************************


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
class WeightedOutcomeAction : public OutcomeManagement{

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
    pair<OutcomeId,prec_t> maximal(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the minimal outcome distribution constraints on the nature's distribution
    Template argument nature represents the function used to select the constrained distribution
    over the outcomes.
    Does not work when the number of outcomes is zero.
    \param valuefunction Value function reference
    \param discount Discount factor
    \return Outcome distribution and the mean value for the minimal bounded solution
     */
    pair<OutcomeId,prec_t> minimal(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the average outcome using a uniform distribution.
    \param valuefunction Updated value function
    \param discount Discount factor
    \return Mean value of the action
     */
    prec_t average(numvec const& valuefunction, prec_t discount) const;

    /**
    Computes the action value for a fixed index outcome.
    \param valuefunction Updated value function
    \param discount Discount factor
    \param index Index of the outcome used
    \return Value of the action
     */
    prec_t fixed(numvec const& valuefunction, prec_t discount, OutcomeId dist) const;

    /**
    Adds a sufficient number of empty outcomes for the outcomeid to be a valid identifier.
    This override also handles properly resizing the distribution.

    The baseline distribution value for the new outcome(s) are set to be:
        1/(new_outcomeid+1)
    That is, it assumes uniform distribution of the outcomes.
    Weights for existing outcomes are scaled appropriately to sum to a value
    that would be equal to a sum of uniformly distributed values.
    */
    Transition& create_outcome(long outcomeid) override;

    /**
    Sets the base distribution over the outcomes.

    The function check for correctness of the distribution.

    \param distribution New distribution of outcomes.
     */
    void set_distribution(const numvec& distribution);

    /**
    Sets weight for a particular outcome.

    The function *does not* check for correctness of the distribution.

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
    Checks whether the outcome distribution is normalized.
    */
    bool is_distribution_normalized() const;

    /**
    Sets an initial uniform value for the threshold (0) and the distribution.
    If the distribution already exists, then it is overwritten.
    */
    void uniform_distribution();

    /** Returns threshold value */
    prec_t get_threshold() const {return threshold;};

    /** Sets threshold value */
    void set_threshold(prec_t threshold){this->threshold = threshold; }

    /** Appends a string representation to the argument */
    void to_string(string& result) const {
        result.append(std::to_string(get_outcomes().size()));
        result.append(" / ");
        result.append(std::to_string(get_distribution().size()));
    }

    /** Whether the provided outcome is valid */
    bool is_outcome_correct(OutcomeId oid) const
        {return (oid.size() == outcomes.size());};

    /** Returns the mean reward from the transition. */
    prec_t mean_reward(OutcomeId outcomedist) const;

    /** Returns the mean transition probabilities */
    Transition mean_transition(OutcomeId outcomedist) const;
};

/// **************************************************************************************
///  L1 Outcome Action
/// **************************************************************************************

typedef WeightedOutcomeAction<worstcase_l1> L1OutcomeAction;

}


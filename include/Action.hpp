#pragma once

#include "definitions.hpp"
#include "Transition.hpp"

#include "cpp11-range-master/range.hpp"
#include <utility>
#include <vector>
#include <limits>
#include <cassert>
#include <string>
#include <numeric>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace craam {

using namespace std;
using namespace util::lang;

// **************************************************************************************
// *** Regular action
// **************************************************************************************

/**
Action in a regular MDP. There is no uncertainty and
the action contains only a single outcome.

An action can be invalid, in which case it is skipped during any computations
and cannot be used during a simulation. See is_valid.
Actions are constructed as valid by default.
*/
class RegularAction{
protected:
    /// Transition probabilities
    Transition outcome;

    /// Invalid actions are skipped during computation
    bool valid = true;

public:
    /** Type of an identifier for an outcome. It is ignored for the simple action. */
    typedef long OutcomeId;

    /** Creates an empty action. */
    RegularAction(){};

    /** Initializes outcomes to the provided transition vector */
    RegularAction(const Transition& outcome) : outcome(outcome) {};

    /** Returns the outcomes. */
    vector<Transition> get_outcomes() const {return vector<Transition>{outcome};};

    /** Returns the single outcome. */
    const Transition& get_outcome(long outcomeid) const {assert(outcomeid == 0); return outcome;};

    /** Returns the single outcome. */
    Transition& get_outcome(long outcomeid) {assert(outcomeid == 0);return outcome;};

    /** Returns the outcome */
    const Transition& operator[](long outcomeid) const {return get_outcome(outcomeid);}

    /** Returns the outcome */
    Transition& operator[](long outcomeid) {return get_outcome(outcomeid);}

    /** Returns the single outcome. */
    const Transition& get_outcome() const {return outcome;};

    /** Returns the single outcome. */
    Transition& get_outcome() {return outcome;};

    /**
    Adds a sufficient number of empty outcomes for the outcomeid to be a valid identifier.
    This method does nothing in this action.
    */
    Transition& create_outcome(long outcomeid){assert(outcomeid == 0);return outcome;}

    /** Normalizes transition probabilities */
    void normalize() {outcome.normalize();};

    /** Returns number of outcomes (1). */
    size_t outcome_count() const {return 1;};

    /**
    Returns whether this is a valid action (or only a placeholder).
    Invalid actions cannot be taken and may result from incomplete
    sampling of a domain. They are skipped in the computation of value function.

    The action is considered valid when there are some transitions
    */
    bool is_valid() const{return valid;};

    /// Sets whether the action is valid (see is_valid)
    void set_validity(bool newvalidity){valid = newvalidity;};

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

    /** Returns a json representation of the action
    \param actionid Includes also action id*/
    string to_json(long actionid = -1) const{
        string result{"{"};
        result += "\"actionid\" : ";
        result += std::to_string(actionid);
        result += ",\"valid\" :";
        result += std::to_string(valid);
        result += ",\"transition\" : ";
        result += outcome.to_json(-1);
        result += "}";
        return result;
    }
};


// **************************************************************************************
//  Outcome Management (a helper class)
// **************************************************************************************

/**
A class that manages creation and access to outcomes to be used by actions.

An action can be invalid, in which case it is skipped during any computations
and cannot be used during a simulation. See is_valid.
Actions are constructed as valid by default.
*/
class OutcomeManagement{

protected:
    /** List of possible outcomes */
    vector<Transition> outcomes;

    /// Invalid actions are skipped during computation
    bool valid = true;
public:
    /** Empty list of outcomes */
    OutcomeManagement() {};

    /** Initializes with a list of outcomes */
    OutcomeManagement(const vector<Transition>& outcomes) : outcomes(outcomes) {};

    /** Empty virtual destructor */
    virtual ~OutcomeManagement() {};

    /**
    Adds a sufficient number of empty outcomes for the outcomeid to be a valid identifier.
    This method is virtual to make overloading safer.
    */
    virtual Transition& create_outcome(long outcomeid){
        if(outcomeid < 0)
            throw invalid_argument("Outcomeid must be non-negative.");

        if(outcomeid >= (long) outcomes.size())
            outcomes.resize(outcomeid + 1);

        return outcomes[outcomeid];
    }

    /**
    Creates a new outcome at the end. Similar to push_back.
    */
    virtual Transition& create_outcome(){return create_outcome(outcomes.size());};

    /** Returns a transition for the outcome. The transition must exist. */
    const Transition& get_outcome(long outcomeid) const {
        assert((outcomeid >= 0l && outcomeid < (long) outcomes.size()));
        return outcomes[outcomeid];};

    /** Returns a transition for the outcome. The transition must exist. */
    Transition& get_outcome(long outcomeid) {
        assert((outcomeid >= 0l && outcomeid < (long) outcomes.size()));
        return outcomes[outcomeid];};

    /** Returns a transition for the outcome. The transition must exist. */
    const Transition& operator[](long outcomeid) const {return get_outcome(outcomeid);}

    /** Returns a transition for the outcome. The transition must exist. */
    Transition& operator[](long outcomeid) {return get_outcome(outcomeid);}

    /** Returns number of outcomes. */
    size_t outcome_count() const {return outcomes.size();};

    /** Returns number of outcomes. */
    size_t size() const {return outcome_count();};

    /** Adds an outcome defined by the transition.
    \param outcomeid Id of the new outcome. Intermediate ids are created empty
    \param t Transition that defines the outcome*/
    void add_outcome(long outcomeid, const Transition& t){ create_outcome(outcomeid) = t; }

    /** Adds an outcome defined by the transition as the last outcome.
    \param t Transition that defines the outcome*/
    void add_outcome(const Transition& t){add_outcome(outcomes.size(), t);};

    /** Returns the list of outcomes */
    const vector<Transition>& get_outcomes() const {return outcomes;};

    /** Normalizes transitions for outcomes */
    void normalize(){
        for(Transition& t : outcomes)
            t.normalize();
    }

    /** Appends a string representation to the argument */
    void to_string(string& result) const{
        result.append(std::to_string(get_outcomes().size()));
    }

    /**
    Returns whether this is a valid action (or only a placeholder).
    Invalid actions cannot be taken and may result from incomplete
    sampling of a domain. They are skipped in the computation of value function.

    The action is considered valid when there are some transitions
    */
    bool is_valid() const{return valid;};

    /// Sets whether the action is valid (see is_valid)
    void set_validity(bool newvalidity){valid = newvalidity;};
};

// **************************************************************************************
//  Discrete Outcome Action
// **************************************************************************************

/**
An action in the robust MDP with discrete outcomes.

An action can be invalid, in which case it is skipped during any computations
and cannot be used during a simulation. See is_valid.
Actions are constructed as valid by default.
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

     /** Whether the provided outcome is valid */
    bool is_outcome_correct(OutcomeId oid) const
        {return (oid >= 0) && ((size_t) oid < outcomes.size());};


    /** Returns the mean reward from the transition. */
    prec_t mean_reward(OutcomeId oid) const { return outcomes[oid].mean_reward();};

    /** Returns the mean transition probabilities */
    Transition mean_transition(OutcomeId oid) const {return outcomes[oid];};

    /** Returns a json representation of action
    \param actionid Includes also action id*/
    string to_json(long actionid = -1) const{
        string result{"{"};
        result += "\"actionid\" : ";
        result += std::to_string(actionid);
        result += ",\"valid\" :";
        result += std::to_string(valid);
        result += ",\"outcomes\" : [";
        for(auto oi : indices(outcomes)){
            const auto& o = outcomes[oi];
            result += o.to_json(oi);
            result += ",";
        }
        if(!outcomes.empty()) result.pop_back(); // remove last comma
        result += "]}";
        return result;
    }

};

// **************************************************************************************
//  Weighted Outcome Action
// **************************************************************************************


/**
An action in a robust MDP in which the outcomes are defined by a weighted function
and a threshold. The uncertain behavior is parametrized by a base distribution
and a threshold value. An example may be a worst case computation:
    \f[ \min \{ u^T v ~:~ \| u - d \|_1 \le  t\} \f]
where \f$ v \f$ are the values for individual outcomes, \f$ d \f$ is the nominal
outcome distribution, and \f$ t \f$ is the threshold.
See L1Action for an example of an instance of this template class.

The distribution d over outcomes is uniform by default:
see WeightedOutcomeAction::create_outcome.

An action can be invalid, in which case it is skipped during any computations
and cannot be used during a simulation. See is_valid.

Actions are constructed as valid by default.
*/
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
    Adds a sufficient number (or 0) of empty outcomes/transitions for the provided outcomeid
    to be a valid identifier. This override also properly resizing the nominal
    outcome distribution and rewighs is accordingly.

    If the corresponding outcome already exists, then it just returns it.

    The baseline distribution value for the new outcome(s) are set to be:
        \f[ d_n' = \frac{1}{n+1}, \f]
    where \f$ n \f$ is the new outcomeid. Weights for existing outcomes (if non-zero) are scaled appropriately to sum to a value
    that would be equal to a sum of uniformly distributed values:
    \f[ d_i' = d_i \frac{m \frac{1}{n+1}}{ \sum_{i=0}^{m} d_i }, \; i = 0 \ldots m \f]
    where \f$ m \f$ is the previously maximal outcomeid; \f$ d_i' \f$ and \f$ d_i \f$ are the new and old weights of the
    outcome \f$ i \f$ respectively. If the outcomes \f$ i < n\f$ do not exist
    they are created with uniform weight.
    This constructs a uniform distribution of the outcomes by default.

    An exception during the computation may leave the distribution in an
    incorrect state.

    \param outcomeid Index of outcome to create
    \returns Transition that corresponds to outcomeid
    */
    Transition& create_outcome(long outcomeid) override{
        if(outcomeid < 0)
            throw invalid_argument("Outcomeid must be non-negative.");
        // 1: compute the weight for the new outcome and old ones

        size_t newsize = outcomeid + 1; // new size of the list of outcomes
        size_t oldsize = outcomes.size(); // current size of the set
        if(newsize <= oldsize){// no need to add anything
            return outcomes[outcomeid];
        }
        // new uniform weight for each element
        prec_t newweight = 1.0/prec_t(outcomeid+1);
        // check if need to scale the existing weights
        if(oldsize > 0){
            auto weightsum = accumulate(distribution.begin(), distribution.end(), 0.0);
            // only scale when the sum is not zero
            if(weightsum > 0){
                prec_t normal = (oldsize * newweight) / weightsum;
                transform(distribution.begin(), distribution.end(),distribution.begin(),
                          [normal](prec_t x){return x * normal;});
            }
        }
        outcomes.resize(newsize);
        // got to resize the distribution too and assign weights that are uniform
        distribution.resize(newsize, newweight);
        return outcomes[outcomeid];
    }

    /**
    Adds a sufficient number of empty outcomes/transitions for the provided outcomeid
    to be a valid identifier. The weights of new outcomes < outcomeid are set
    to 0. This operation does rescale weights in order to preserve their sum.

    If the outcome already exists, its nominal weight is overwritten.

    Note that this operation may leave the action in an invalid state in
    which the nominal outcome distribution does not sum to 1.

    \param outcomeid Index of outcome to create
    \param weight New nominal weight for the outcome.
    \returns Transition that corresponds to outcomeid
    */
    Transition& create_outcome(long outcomeid, prec_t weight){
        if(outcomeid < 0)
            throw invalid_argument("Outcomeid must be non-negative.");
        assert(weight >= 0 && weight <= 1);

        if(outcomeid >= static_cast<long>(outcomes.size())){ // needs to resize arrays
            outcomes.resize(outcomeid+1);
            distribution.resize(outcomeid+1);
        }
        set_distribution(outcomeid, weight);
        return outcomes[outcomeid];
    }

    /**
    Sets the base distribution over the outcomes.

    The function check for correctness of the distribution.

    \param distribution New distribution of outcomes.
     */
    void set_distribution(const numvec& distribution){
        if(distribution.size() != outcomes.size())
            throw invalid_argument("Invalid distribution size.");
        prec_t sum = accumulate(distribution.begin(),distribution.end(), 0.0);
        if(sum < 0.99 || sum > 1.001)
            throw invalid_argument("Distribution does not sum to 1.");
        if((*min_element(distribution.begin(),distribution.end())) < 0)
            throw invalid_argument("Distribution must be non-negative.");

        this->distribution = distribution;
    }

    /**
    Sets weight for a particular outcome.

    The function *does not* check for correctness of the distribution.

    \param distribution New distribution of outcomes.
    \param weight New weight
     */
    void set_distribution(long outcomeid, prec_t weight){
        assert(outcomeid >= 0 && (size_t) outcomeid < outcomes.size());
        distribution[outcomeid] = weight;
    }

    /** Returns the baseline distribution over outcomes. */
    const numvec& get_distribution() const {return distribution;};

    /**
    Normalizes outcome weights to sum to one. Assumes that the distribution
    is initialized. Exception is thrown if the distribution sums
    to zero.
    */
    void normalize_distribution(){
        auto weightsum = accumulate(distribution.begin(), distribution.end(), 0.0);

        if(weightsum > 0.0){
            for(auto& p : distribution)
                p /= weightsum;
        }else{
            throw invalid_argument("Distribution sums to 0 and cannot be normalized.");
        }
    }

    /**
    Checks whether the outcome distribution is normalized.
    */
    bool is_distribution_normalized() const{
        return abs(1.0-accumulate(distribution.begin(), distribution.end(), 0.0)) < SOLPREC;
    }

    /**
    Sets an initial uniform value for the threshold (0) and the distribution.
    If the distribution already exists, then it is overwritten.
    */
    void uniform_distribution(){
        distribution.clear();
        if(outcomes.size() > 0)
            distribution.resize(outcomes.size(), 1.0/ (prec_t) outcomes.size());
        threshold = 0.0;
    }

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
    prec_t mean_reward(OutcomeId outcomedist) const{
        assert(outcomedist.size() == outcomes.size());
        prec_t result = 0;
        for(size_t i = 0; i < outcomes.size(); i++){
            result += outcomedist[i] * outcomes[i].mean_reward();
        }
        return result;
    }

    /** Returns the mean transition probabilities */
    Transition mean_transition(OutcomeId outcomedist) const{
        assert(outcomedist.size() == outcomes.size());
        Transition result;
        for(size_t i = 0; i < outcomes.size(); i++)
            outcomes[i].probabilities_addto(outcomedist[i], result);
        return result;
    }

    /** Returns a json representation of action
    \param actionid Includes also action id*/
    string to_json(long actionid = -1) const{
        string result{"{"};
        result += "\"actionid\" : ";
        result += std::to_string(actionid);
        result += ",\"valid\" :";
        result += std::to_string(valid);
        result += ",\"threshold\" : ";
        result += std::to_string(threshold);
        result += ",\"outcomes\" : [";
        for(auto oi : indices(outcomes)){
            const auto& o = outcomes[oi];
            result +=o.to_json(oi);
            result +=",";
        }
        if(!outcomes.empty()) result.pop_back(); // remove last comma
        result += "],\"distribution\" : [";
        for(auto d : distribution){
            result += std::to_string(d);
            result += ",";
        }
        if(!distribution.empty()) result.pop_back(); // remove last comma
        result += "]}";
        return result;
    }
};

}




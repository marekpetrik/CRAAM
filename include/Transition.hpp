#pragma once

#include<vector>

#include "definitions.hpp"

using namespace std;

namespace craam {

const prec_t tolerance = 1e-5;

/**
  Represents sparse transition probabilities and rewards from a single state.

  The destination indexes are sorted increasingly (as added). This makes it simpler to
  aggregate multiple transition probabilities and should also make value iteration
  more cache friendly. However, transitions need to be added with increasing IDs to
  prevent excessive performance degradation.
 */
class Transition {

public:
    Transition(){};

    /**
    Creates a single transition from raw data.

    Because the transition indexes are stored increasingly sorted, this method
    must sort (and aggregate duplicate) the indices.

    \param indices The indexes of states to transition to
    \param probabilities The probabilities of transitions
    \param rewards The associated rewards with each transition
    */
    Transition(const indvec& indices,
                const numvec& probabilities,
                const numvec& rewards);

    /**
    Creates a single transition from raw data with uniformly zero rewards.

    Because the transition indexes are stored increasingly sorted, this method
    must sort (and aggregate duplicate) the indices.

    \param indices The indexes of states to transition to
    \param probabilities The probabilities of transitions
    */
    Transition(const indvec& indices,
                const numvec& probabilities);

    /**
    Creates a single transition from raw data with uniformly zero rewards,
    where destination states are indexed automatically starting with 0.

    \param probabilities The probabilities of transitions; indexes are implicit.
    */
    Transition(const numvec& probabilities);

    /**
    Adds a single transitions probability to the existing probabilities.

    If the transition to the desired state already exists, then the transition
    probability is added and the reward is updated as a weighted combination.

    Transition probabilities are not checked to sum to one.

    \param stateid ID of the target state
    \param probability Probability of transitioning to this state
    \param reward The reward associated with the transition
     */
    void add_sample(long stateid, prec_t probability, prec_t reward);

    prec_t sum_probabilities() const;
    /**
    Normalizes the probabilities to sum to 1. Exception is thrown if the
    distribution sums to 0.
    */
    void normalize();
    bool is_normalized() const;

    /**
    Computes value for the transition and a value function.

    When there are no target states, the function terminates with an error.

    \param valuefunction Value function, or an arbitrary vector of values
    \param discount Discount factor, optional (default value 1)
     */
    prec_t compute_value(numvec const& valuefunction, prec_t discount = 1.0) const;

    /** Computes the mean return from this transition */
    prec_t mean_reward() const;

    /** Returns the number of target states with non-zero transition probabilities.  */
    size_t size() const {return indices.size();};

    /**
    Returns the maximal indexes involved in the transition.
    Returns -1 for and empty transition.
    */
    long max_index() const {return indices.empty() ? -1 : indices.back();};

    /**
    Constructs and returns a dense vector of probabilities.
    \param size Size of the constructed vector
    */
    numvec probabilities_vector(size_t size) const;

    /**
    Scales transition probabilities according to the provided parameter
    and adds them to the provided vector.
    \param scale Multiplicative modification of transition probabilities
    \param transition Transition probabilities being added to. This value
                        is modified within the function.
    */
    void probabilities_addto(prec_t scale, numvec& transition) const;

    const indvec& get_indices() const {return indices;};
    const numvec& get_probabilities() const {return probabilities;};
    const numvec& get_rewards() const {return rewards;};

    void set_reward(long sampleid, prec_t reward) {rewards[sampleid] = reward;};
    prec_t get_reward(long sampleid) const {return rewards[sampleid];};

protected:

    indvec indices;
    numvec probabilities;
    numvec rewards;
};

}

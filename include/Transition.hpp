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

    Transition(const vector<long>& indices, 
                const numvec& probabilities, 
                const numvec& rewards);

    void add_sample(long stateid, prec_t probability, prec_t reward);

    prec_t sum_probabilities() const;
    void normalize();
    bool is_normalized() const;

    prec_t compute_value(numvec const& valuefunction, prec_t discount = 1.0) const;
    prec_t mean_reward() const;

    size_t size() const{
        /** Returns the number of target states with non-zero transition
         probabilities.  */
        return indices.size();
    }

    long max_index() const{
        /** Returns the maximal indexes involved in the transition.  */
        return indices.back();
    }

    // probability manipulation
    numvec probabilities_vector(size_t size) const;
    void probabilities_addto(prec_t scale, numvec& transition) const;

    const vector<long>& get_indices() const {return indices;};
    const numvec& get_probabilities() const {return probabilities;};
    const numvec& get_rewards() const {return rewards;};

    void set_reward(long sampleid, prec_t reward) {rewards[sampleid] = reward;};
    prec_t get_reward(long sampleid) const {return rewards[sampleid];};

protected:
    vector<long> indices;
    numvec probabilities;
    numvec rewards;
};

}

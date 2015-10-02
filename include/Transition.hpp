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
    vector<long> indices;
    vector<prec_t> probabilities;
    vector<prec_t> rewards;

    Transition(){};

    Transition(const vector<long>& indices, const vector<prec_t>& probabilities, 
                const vector<prec_t>& rewards);

    void add_sample(long stateid, prec_t probability, prec_t reward);

    prec_t sum_probabilities() const;
    void normalize();
    bool is_normalized() const;

    prec_t compute_value(vector<prec_t> const& valuefunction, prec_t discount) const;

    size_t size() const{
        return indices.size();
    }

    long max_index() const{
        /** Returns the maximal indices involved in the transition.  */
        return indices.back();
    }
};

}

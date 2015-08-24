#ifndef TRANSITION_H
#define TRANSITION_H

#include<vector>

#include "definitions.hpp"

using namespace std;

namespace craam {

const prec_t tolerance = 1e-5;

class Transition {
/**
 * \brief Represents sparse transition probabilities and rewards from a single state.
 *
 * The destination indexes are sorted increasingly. This makes it simpler to
 * aggregate multiple transition probabilities and should also make value iteration
 * more cache friendly. However, transitions need to be added with increasing IDs to
 * prevent excessive performance degradation.
 *
 */

public:
    vector<long> indices;
    vector<prec_t> probabilities;
    vector<prec_t> rewards;

    Transition(){};

    Transition(vector<long> indices, vector<prec_t> probabilities, vector<prec_t> rewards);

    void add_sample(long stateid, prec_t probability, prec_t reward);

    prec_t sum_probabilities() const;
    void normalize();
    bool is_normalized() const;

    prec_t compute_value(vector<prec_t> const& valuefunction, prec_t discount) const;
};

}

#endif // TRANSITION_H

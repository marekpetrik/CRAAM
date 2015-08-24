#include <algorithm>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>
#include <assert.h>

#include "Transition.hpp"
#include "definitions.hpp"

namespace craam {

Transition::Transition(vector<long> const indices, vector<prec_t> probabilities, vector<prec_t> rewards){
    /** \brief Creates a single transition
     *
     * \param indexes The indexes of states to transition to
     * \param probabilities The probabilities of transitions
     * \param rewards The associated rewards
     * \return
     */

    this->indices = indices;
    this->rewards = rewards;
    this->probabilities = probabilities;
}

void Transition::add_sample(long stateid, prec_t probability, prec_t reward) {
    /**
     * \brief Adds a single transitions probability to the existing probabilities.
     *
     * If the transition to the desired state already exists, then the transition
     * probability is added and the reward is updated as a weighted combination.
     *
     * Transition probabilities are not checked to sum to one.
     *
     * \param stateid ID of the target state
     * \param probability Probability of transitioning to this state
     * \param reward The reward associated with the transition
     */

    if(probability < -0.001) throw invalid_argument("probabilities must be non-negative.");
    if(stateid < 0) throw invalid_argument("invalid stateid");


    // test for the last index; the index is not in the transition yet and belong to the end
    if(indices.size() == 0 || this->indices.back() < stateid){
        indices.push_back(stateid);
        probabilities.push_back(probability);
        rewards.push_back(reward);
    }
    else{ // the index is already in the transitions, or belongs in the middle

        size_t findex;  // lower bound on the index of the element
        bool present;   // whether the index was found

        // test the last element for efficiency sake
        if(stateid == indices.back()){
            findex = indices.size() - 1;
            present = true;
        }
        else{
            // find the closest existing index to the
            auto fiter = lower_bound(indices.begin(),indices.end(),stateid);
            findex = fiter - indices.begin();
            present = (*fiter == stateid);
        }

        if(present){    // there is a transition to this element already
            auto p_old = probabilities[findex];
            auto r_old = rewards[findex];
            auto new_reward = (p_old * r_old + probability * reward) / (p_old + probability);

            probabilities[findex] += probability;
            rewards[findex] = new_reward;
        }else{          // the transition is not there, the element needs to be inserted
            indices.insert(indices.begin()+findex,stateid);
            probabilities.insert(probabilities.begin()+findex,probability);
            rewards.insert(rewards.begin()+findex,reward);
        }
    }
}

prec_t Transition::sum_probabilities() const{
    return accumulate(probabilities.begin(),probabilities.end(),0.0);
}

bool Transition::is_normalized() const{
    return abs(1.0 - sum_probabilities()) < tolerance;
}

void Transition::normalize(){

    prec_t sp = sum_probabilities();

    if(sp != 0.0){
        for (auto& p : probabilities){
            p /= sp;
        }
    }
}

prec_t Transition::compute_value(vector<prec_t> const& valuefunction, prec_t discount) const{
    /**
     * \brief Computes value for the transition and a value function.
     *
     * When there are no target states, the function terminates with an error.
     */

    auto count = indices.size();

    if(count == 0){
        throw range_error("no transitions defined");
    }

    prec_t value = (prec_t) 0.0;

    for(size_t c = 0; c < count; c++){
        value +=  probabilities[c] * (rewards[c] + discount * valuefunction[indices[c]]);
    }
    return value;
}

}

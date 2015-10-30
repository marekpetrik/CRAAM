#include <algorithm>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>
#include <assert.h>

#include "Transition.hpp"
#include "definitions.hpp"

namespace craam {


Transition::Transition(const indvec& indices, const vector<prec_t>& probabilities, 
                        const vector<prec_t>& rewards){
    /** 
        Creates a single transition from raw data. 

        Because the transition indexes are stored increasingly sorted, this method
        must sort (and aggregate duplicate) the indices. 

        \param indices The indexes of states to transition to
        \param probabilities The probabilities of transitions
        \param rewards The associated rewards with each transition

     */


    if(indices.size() != probabilities.size() || indices.size() != rewards.size())
        throw invalid_argument("All parameters for the constructor of Transition must have the same size.");

    auto sorted = sort_indexes(indices);

    for(auto&& k : sorted)
        add_sample(indices[k],probabilities[k],rewards[k]);
}

Transition::Transition(const indvec& indices, const vector<prec_t>& probabilities){
    /** 
        Creates a single transition from raw data with uniformly zero rewards. 

        Because the transition indexes are stored increasingly sorted, this method
        must sort (and aggregate duplicate) the indices. 

        \param indices The indexes of states to transition to
        \param probabilities The probabilities of transitions

     */

    if(indices.size() != probabilities.size() || indices.size() != rewards.size())
        throw invalid_argument("All parameters for the constructor of Transition must have the same size.");

    auto sorted = sort_indexes(indices);

    for(auto&& k : sorted)
        add_sample(indices[k],probabilities[k],0.0);
}


void Transition::add_sample(long stateid, prec_t probability, prec_t reward) {
    /**
      Adds a single transitions probability to the existing probabilities.

      If the transition to the desired state already exists, then the transition
      probability is added and the reward is updated as a weighted combination.

      Transition probabilities are not checked to sum to one.

      \param stateid ID of the target state
      \param probability Probability of transitioning to this state
      \param reward The reward associated with the transition
     */

    if(probability < -0.001) throw invalid_argument("probabilities must be non-negative.");
    if(stateid < 0) throw invalid_argument("State id must be non-negative.");

    // if the probability is 0, just do not add it
    if(probability <= 0){
        return;
    }

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
        for (auto& p : probabilities)
            p /= sp;
    }
}

prec_t Transition::compute_value(vector<prec_t> const& valuefunction, prec_t discount) const{
    /**
      Computes value for the transition and a value function.

      When there are no target states, the function terminates with an error.

      \param valuefunction Value function, or an arbitrary vector of values
      \param discount Discount factor, optional (default value 1)
     */

    auto scount = indices.size();

    //TODO: check how much complexity these statements are adding
    if(scount == 0)
        throw range_error("No transitions defined.");

    prec_t value = 0.0;

    for(size_t c = 0; c < scount; c++){
        value +=  probabilities[c] * (rewards[c] + discount * valuefunction[indices[c]]);
    }
    return value;
}

prec_t Transition::mean_reward() const{
    /**
      Computes the mean return from this transition
     */

    auto scount = indices.size();

    if(scount == 0)
        throw range_error("No transitions defined.");

    prec_t value = 0.0;

    for(size_t c = 0; c < scount; c++){
        value +=  probabilities[c] * rewards[c];
    }
    return value;
}

vector<prec_t> Transition::probabilities_vector(size_t size) const{
    /**
        Constructs and returns a dense vector of probabilities.

        \param size Size of the constructed vector
     */
    vector<prec_t> result(size, 0.0);

    for(size_t i = 0; i < this->size(); i++){
        result[indices[i]] = probabilities[i];
    }

    return result;
}
void Transition::probabilities_addto(prec_t scale, vector<prec_t>& transition) const{
    /** 
        Scales transition probabilities and adds them to the provided vector.

        \param scale Multiplicative modification of transition probabilities
        \param transition Transition probabilities being added to. This value
                            is modified within the function.
     */

    for(size_t i = 0; i < size(); i++){
        transition[indices[i]] += scale*probabilities[i];
    }
}

}   


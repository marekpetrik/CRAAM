#include "definitions.hpp"
#include "Transition.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>
#include <numeric>
#include <cmath>
#include <assert.h>

#include "cpp11-range-master/range.hpp"

using namespace util::lang;

namespace craam {

Transition::Transition(const indvec& indices, const numvec& probabilities,
                        const numvec& rewards){

    if(indices.size() != probabilities.size() || indices.size() != rewards.size())
        throw invalid_argument("All parameters for the constructor of Transition must have the same size.");

    auto sorted = sort_indexes(indices);

    for(auto&& k : sorted)
        add_sample(indices[k],probabilities[k],rewards[k]);
}

Transition::Transition(const indvec& indices, const numvec& probabilities){

    if(indices.size() != probabilities.size())
        throw invalid_argument("All parameters for the constructor of Transition must have the same size.");

    auto sorted = sort_indexes(indices);

    for(auto k : sorted)
        add_sample(indices[k],probabilities[k],0.0);
}


Transition::Transition(const numvec& probabilities){

    for(auto k : range((size_t)0, probabilities.size()))
        add_sample(k, probabilities[k], 0.0);
}

void Transition::add_sample(long stateid, prec_t probability, prec_t reward) {

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
    if(indices.empty())
        return true;
    else
        return abs(1.0 - sum_probabilities()) < tolerance;
}

void Transition::normalize(){
    prec_t sp = sum_probabilities();

    if(sp != 0.0){
        for(auto& p : probabilities)
            p /= sp;
    }else{
        throw invalid_argument("Probabilities sum to 0 and cannot be normalized.");
    }
}

prec_t Transition::compute_value(numvec const& valuefunction, prec_t discount) const{
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
    auto scount = indices.size();

    if(scount == 0)
        throw range_error("No transitions defined.");

    prec_t value = 0.0;

    for(size_t c = 0; c < scount; c++){
        value +=  probabilities[c] * rewards[c];
    }
    return value;
}

numvec Transition::probabilities_vector(size_t size) const{
    numvec result(size, 0.0);

    for(size_t i = 0; i < this->size(); i++){
        result[indices[i]] = probabilities[i];
    }

    return result;
}

void Transition::probabilities_addto(prec_t scale, numvec& transition) const{

    for(size_t i = 0; i < size(); i++)
        transition[indices[i]] += scale*probabilities[i];
}

void Transition::probabilities_addto(prec_t scale, Transition& transition) const{

    for(size_t i = 0; i < size(); i++)
        transition.add_sample(indices[i], scale*probabilities[i], 0);
}

}


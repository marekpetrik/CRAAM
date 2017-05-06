#include "definitions.hpp"
#include "Transition.hpp"

#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>

#include "cpp11-range-master/range.hpp"

using namespace util::lang;

namespace craam {

Transition::Transition(const indvec& indices, const numvec& probabilities,
                        const numvec& rewards){
    if(indices.size() != probabilities.size() || indices.size() != rewards.size())
        throw invalid_argument("All parameters for the constructor of Transition must have the same size.");
    auto sorted = sort_indexes(indices);
    for(auto k : sorted)
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
    for(auto k : util::lang::indices(probabilities))
        add_sample(k, probabilities[k], 0.0);
}

void Transition::add_sample(long stateid, prec_t probability, prec_t reward) {

    if(probability < -0.001) throw invalid_argument("probabilities must be non-negative.");
    if(stateid < 0) throw invalid_argument("State id must be non-negative.");
    // if the probability is 0 or negative, just do not add the sample
    if(probability <= 0) return; 

    // test for the last index; the index is not in the transition yet and belong to the end
    if(indices.size() == 0 || this->indices.back() < stateid){
        indices.push_back(stateid);
        probabilities.push_back(probability);
        rewards.push_back(reward);
    }
    // the index is already in the transitions, or belongs in the middle
    else{
        size_t findex;  // lower bound on the index of the element
        bool present;   // whether the index was found

        // test the last element for efficiency sake
        if(stateid == indices.back()){
            findex = indices.size() - 1;
            present = true;
        }
        else{
            // find the closest existing index to the new one
            auto fiter = lower_bound(indices.begin(),indices.end(),stateid);
            findex = fiter - indices.begin();
            present = (*fiter == stateid);
        }
        // there is a transition to this element already
        if(present){
            auto p_old = probabilities[findex];
            probabilities[findex] += probability;
            auto r_old = rewards[findex];
            auto new_reward = (p_old * r_old + probability * reward) / 
                                (probabilities[findex]);
            rewards[findex] = new_reward;
        // the transition is not there, the element needs to be inserted
        }else{
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
    // nothing to do if there are no transitions
    if(probabilities.empty())
        return;

    prec_t sp = sum_probabilities();
    if(sp != 0.0){
        for(auto& p : probabilities)
            p /= sp;
    }else{
        throw invalid_argument("Probabilities sum to 0 and cannot be normalized.");
    }
}

prec_t Transition::compute_value(numvec const& valuefunction, prec_t discount) const{

    if(indices.empty())
        throw range_error("No transitions defined for the state action-pair. Cannot compute value.");

    prec_t value = 0.0;

    for(size_t c : util::lang::indices(indices)){
        value +=  probabilities[c] * (rewards[c] + discount * valuefunction[indices[c]]);
    }
    return value;
}

prec_t Transition::mean_reward() const{

    if(indices.empty())
        throw range_error("No transitions defined. Cannot compute mean reward.");

    prec_t value = 0.0;

    for(size_t c : util::lang::indices(indices)){
        value +=  probabilities[c] * rewards[c];
    }
    return value;
}


void Transition::probabilities_addto(prec_t scale, numvec& transition) const{
    for(size_t i : util::lang::indices(*this))
        transition[indices[i]] += scale*probabilities[i];
}

void Transition::probabilities_addto(prec_t scale, Transition& transition) const{

    for(size_t i : util::lang::indices(*this))
        transition.add_sample(indices[i], scale*probabilities[i], scale*rewards[i]);
}

numvec Transition::probabilities_vector(size_t size) const{
    if(max_index() >= 0 && static_cast<long>(size) <= max_index())
        throw range_error("Size must be greater than the maximal index");

    numvec result(size, 0.0);

    for(size_t i : util::lang::indices(indices)){
        result[indices[i]] = probabilities[i];
    }

    return result;
}

numvec Transition::rewards_vector(size_t size) const{
    if(max_index() >= 0 && static_cast<long>(size) <= max_index())
        throw range_error("Size must be greater than the maximal index");

    numvec result(size, 0.0);

    for(size_t i : util::lang::indices(indices)){
        result[indices[i]] = rewards[i];
    }

    return result;
}

string Transition::to_json(long outcomeid) const{
    string result{"{"};
    result += "\"outcomeid\" : ";
    result += std::to_string(outcomeid);
    result += ",\"stateids\" : [";
    for(auto i : indices){
        result += std::to_string(i);
        result += ",";
    }
    if(!indices.empty()) result.pop_back();// remove last comma
    result += "],\"probabilities\" : [";
    for(auto p : probabilities){
        result += std::to_string(p);
        result += ",";
    }
    if(!probabilities.empty()) result.pop_back();// remove last comma
    result += "],\"rewards\" : [" ;
    for(auto r : rewards){
        result += std::to_string(r);
        result += ",";
    }
    if(!rewards.empty()) result.pop_back();// remove last comma
    result += "]}";
    return result;
}

}


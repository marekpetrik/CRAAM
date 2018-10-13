// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "definitions.hpp"

#include <rm/range.hpp>
#include<vector>
#include<string>
#include <algorithm>
#include <stdexcept>
#include <numeric>
#include <cmath>


namespace craam {

using namespace std;
using namespace util::lang;

/** tolerance for checking whether a transition probability is normalized */
const prec_t tolerance = 1e-5;

/**
  Represents sparse transition probabilities and rewards from a single state.
  The class can be also used to represent a generic sparse distribution.

  The destination indexes are sorted increasingly (as added). This makes it simpler to
  aggregate multiple transition probabilities and should also make value iteration
  more cache friendly. However, transitions need to be added with increasing IDs to
  prevent excessive performance degradation.
 */
class Transition {

public:
    Transition() : indices(0), probabilities(0), rewards(0) {}

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
                const numvec& rewards) : Transition() {

        if(indices.size() != probabilities.size() || indices.size() != rewards.size())
            throw invalid_argument("All parameters for the constructor of Transition must have the same size.");
        auto sorted = sort_indexes(indices);
        for(auto k : sorted)
            add_sample(indices[k],probabilities[k],rewards[k]);
    }

    /**
    Creates a single transition from raw data with uniformly zero rewards.

    Because the transition indexes are stored increasingly sorted, this method
    must sort (and aggregate duplicate) the indices.

    \param indices The indexes of states to transition to
    \param probabilities The probabilities of transitions
    */
    Transition(const indvec& indices,
                const numvec& probabilities) : Transition() {

        if(indices.size() != probabilities.size())
            throw invalid_argument("All parameters for the constructor of Transition must have the same size.");
        auto sorted = sort_indexes(indices);
        for(auto k : sorted)
            add_sample(indices[k],probabilities[k],0.0);
    }

    /**
    Creates a single transition from raw data with uniformly zero rewards,
    where destination states are indexed automatically starting with 0.

    \param probabilities The probabilities of transitions; indexes are implicit.
    */
    Transition(const numvec& probabilities) : Transition() {
        for(auto k : util::lang::indices(probabilities))
            add_sample(long(k), probabilities[k], 0.0);
    }

    /**
    Adds a single transitions probability to the existing probabilities.

    If the transition to a state does not exist, then it is simply added to the
    list. If the transition to the desired state already exists, then the transition
    probability is added and the reward is updated as a weighted combination.
    Let \f$ p(s) \f$ and \f$ r(s) \f$ be the current transition probability
    and reward respectively. The updated transition probability and reward are:
        - Probability:
            \f[ p'(s) = p(s) + p \f]
        - Reward:
            \f[ r'(s) = \frac{p(s) \, r(s) + p \, r}{p'(s)} \f]
    Here, \f$ p \f$ is the argument probability and \f$ r \f$ is the argument
    reward.

    When the function is called multiple times with \f$ p_1 \ldots p_n \f$ and
    \f$  r_1 \ldots r_n \f$ for a single \f$ s \f$ then:
        - Probability:
            \f[ p'(s) = \sum_{i=1}^{n} p_i \f]
        - Reward:
            \f[ r'(s) = \frac{  \sum_{i=1}^{n} p_i \, r_i}{p'(s)} \f]


    Transition probabilities are not checked to sum to one.

    \param stateid ID of the target state
    \param probability Probability of transitioning to this state
    \param reward The reward associated with the transition
     */
    void add_sample(long stateid, prec_t probability, prec_t reward){

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
                auto fiter = lower_bound(indices.cbegin(),indices.cend(),stateid);
                findex = fiter - indices.cbegin();
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
                indices.insert(indices.cbegin() + findex, stateid);
                probabilities.insert(probabilities.cbegin() + findex,probability);
                rewards.insert(rewards.cbegin() + findex,reward);
            }
        }

    }

    prec_t sum_probabilities() const{
        return accumulate(probabilities.cbegin(),probabilities.cend(),0.0);
    }

    /**
    Normalizes the transition probabilities to sum to 1. Exception is thrown if the
    distribution sums to 0.
    */
    void normalize(){
        // nothing to do if there are no transitions
        if(probabilities.empty())
            return;

        prec_t sp = sum_probabilities();
        if(sp > tolerance){
            for(auto& p : probabilities)
                p /= sp;
        }else{
            throw invalid_argument("Probabilities sum to 0 (or close) and cannot be normalized.");
        }
    }

    /** \returns Whether the transition probabilities sum to 1. */
    bool is_normalized() const{
        if(indices.empty()) return true;
        else return abs(1.0 - sum_probabilities()) < tolerance;
    }

    /**
    Computes value for the transition and a value function.

    When there are no target states, the function terminates with an error.

    \param valuefunction Value function, or an arbitrary vector of values
    \param discount Discount factor, optional (default value 1)
    \param probabilities Custom probability distribution. It must be of the same length as
                the number of *nonzero* transition probabilities. The length is NOT checked
                in a release build.
     */
    prec_t value(numvec const& valuefunction, prec_t discount, numvec probabilities) const{
        assert(valuefunction.size() >= probabilities.size());
        assert(rewards.size() == probabilities.size());
        assert(probabilities.size() == indices.size());

        if(indices.empty())
            throw range_error("No transitions defined for the state action-pair. Cannot compute value.");
        prec_t value = 0.0;

        //Note: in simple benchmarks, the simd statement seems to speed up the computation
        // by a factor of 2-4 with -march=native on a computer with AVX support
        #pragma omp simd reduction(+:value)
        for(size_t c = 0; c < size(); c++){
            value +=  probabilities[c] * (rewards[c] + discount * valuefunction[indices[c]]);
        }
        return value;
    }

    /**
    Computes value for the transition and a value function.

    When there are no target states, the function terminates with an error.

    \param valuefunction Value function, or an arbitrary vector of values
    \param discount Discount factor, optional (default value 1)
     */
    prec_t value(numvec const& valuefunction, prec_t discount = 1.0) const{

        return value(valuefunction, discount, probabilities);
    }

    /** Computes the mean return from this transition with custom transition probabilities */
    prec_t mean_reward(const numvec& probabilities) const{
        assert(probabilities.size() == size());
        if(indices.empty())
            throw range_error("No transitions defined. Cannot compute mean reward.");

        return inner_product(probabilities.cbegin(), probabilities.cend(), rewards.cbegin(), 0.0);
    }

    /** Computes the mean return from this transition */
    prec_t mean_reward() const{
        return mean_reward(probabilities);
    }

    /** Returns the number of target states with non-zero transition probabilities.  */
    size_t size() const {
        return indices.size();
    }

    /** Checks if the transition is empty. */
    bool empty() const {
        return indices.empty();
    }

    /**
    Returns the maximal indexes involved in the transition.
    Returns -1 for and empty transition.
    */
    long max_index() const {
        return indices.empty() ? -1 : indices.back();
    }

    /**
    Scales transition probabilities according to the provided parameter
    and adds them to the provided vector. This method ignores rewards.
    \param scale Multiplicative modification of transition probabilities
    \param transition Transition probabilities being added to. This value
                        is modified within the function.
    */
    void probabilities_addto(prec_t scale, numvec& transition) const{
        for(size_t i : util::lang::indices(*this))
            transition[indices[i]] += scale * probabilities[i];
    }

    /**
    Scales transition probabilities and rewards according to the provided parameter
    and adds them to the provided vector.

    \param scale Multiplicative modification of transition probabilities
    \param transition Transition probabilities being added to. This value
                        is modified within the function.
    */
    void probabilities_addto(prec_t scale, Transition& transition) const{
        for(size_t i : util::lang::indices(*this))
            transition.add_sample(indices[i], scale*probabilities[i], scale*rewards[i]);
    }

    /**
    Constructs and returns a dense vector of probabilities, which
    includes 0 transition probabilities.
    \param size Size of the constructed vector
    */
    numvec probabilities_vector(size_t size) const{

        if(max_index() >= 0 && static_cast<long>(size) <= max_index())
            throw range_error("Size must be greater than the maximal index");
        numvec result(size, 0.0);
        for(size_t i : util::lang::indices(indices))
            result[indices[i]] = probabilities[i];
        return result;
    }

    /**
    Constructs and returns a dense vector of rewards, which
    includes 0 transition probabilities. Rewards for indices with
    zero transition probability are zero.
    \param size Size of the constructed vector
    */
    numvec rewards_vector(size_t size) const{

        if(max_index() >= 0 && static_cast<long>(size) <= max_index())
            throw range_error("Size must be greater than the maximal index");
        numvec result(size, 0.0);
        for(size_t i : util::lang::indices(indices))
            result[indices[i]] = rewards[i];
        return result;
    }

    /** Indices with positive probabilities.  */
    const indvec& get_indices() const {return indices;};

    /** Index of the k-th state with non-zero probability */
    long get_index(long k){assert(k>=0 && k < long(size())); return indices[k];}

    /**
    Returns list of positive probabilities for indexes returned by
    get_indices. See also probabilities_vector.
    */
    const numvec& get_probabilities() const {return probabilities;};

    void set_probabilities(numvec probabilities){
        this->probabilities = probabilities;
    }

    /**
    Rewards for indices with positive probabilities returned by
    get_indices. See also rewards_vector.
    */
    const numvec& get_rewards() const {return rewards;};

    /** Sets the reward for a transition to a particular state */
    void set_reward(long sampleid, prec_t reward) {rewards[sampleid] = reward;};

    /** Gets the reward for a transition to a particular state */
    prec_t get_reward(long sampleid) const {
        assert(sampleid >= 0 && sampleid < long(size())); 
        return rewards[sampleid];
    };

    /** Returns a json representation of transition probabilities
    \param outcomeid Includes also outcome id*/
    string to_json(long outcomeid = -1) const{
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

protected:

    /// List of state indices
    indvec indices;
    /// List of probability distributions to states
    numvec probabilities;
    /// List of rewards associated with transitions
    numvec rewards;
};

}

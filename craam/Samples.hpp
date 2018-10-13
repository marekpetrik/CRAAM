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
#include "RMDP.hpp"
#include "modeltools.hpp"

#include <rm/range.hpp>
#include <set>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cassert>
#include <utility>
#include <vector>
#include <string>


namespace craam{

/// A namespace for handling sampling and simulation
namespace msen{

using namespace util::lang;
using namespace std;


/**
Represents a single transition between two states after taking an action:
 \f[ (s, a, s', r, w) \f]
where:
    - \f$ s \f$ is the originating state
    - \f$ a \f$ is the action taken
    - \f$ s' \f$ is the target state
    - \f$ r \f$ is the reward
    - \f$ w \f$ is the weight (or importance) of the state. It is
                like probability, except it does not have to sum to 1.
                It must be non-negative.

In addition, the sample also includes step and the run. These are
used for example to compute the return from samples.

\tparam State MDP state: \f$ s, s'\f$
\tparam Action MDP action: \f$ a \f$
 */
template <class State, class Action>
class Sample {
public:
    Sample(State state_from, Action action, State state_to,
           prec_t reward, prec_t weight, long step, long run):
        _state_from(move(state_from)), _action(move(action)),
        _state_to(move(state_to)), _reward(reward), _weight(weight), _step(step), _run(run){
        assert(weight >= 0);};

    /** Original state */
    State state_from() const {return _state_from;};
    /** Action taken */
    Action action() const {return _action;};
    /** Destination state */
    State state_to() const {return _state_to;};
    /** Reward associated with the sample */
    prec_t reward() const {return _reward;};
    /// Sample weight
    prec_t weight() const {return _weight;};
    /// Number of the step in an one execution of the simulation
    long step() const {return _step;};
    /// Number of the actual execution
    long run() const {return _run;};

protected:
    /// Original state
    State _state_from;
    /// Action taken
    Action _action;
    /// Destination state
    State _state_to;
    /// Reward associated with the sample
    prec_t _reward;
    /// Sample weight
    prec_t _weight;
    /// Number of the step in an one execution of the simulation
    long _step;
    /// Number of the actual execution
    long _run;
};


/**
General representation of samples:
\f[ \Sigma = (s_i, a_i, s_i', r_i, w_i)_{i=0}^{m-1} \f]
See Sample for definitions of individual values.

\tparam State Type defining states
\tparam Action Type defining actions
 */
template <class State, class Action>
class Samples {
public:

   Samples(): states_from(), actions(), states_to(), rewards(), cumulative_rewards(), weights(), runs(), steps(), initial() {};
    

    /** Adds an initial state */
    void add_initial(const State& decstate){
        this->initial.push_back(decstate);
    };

    /** Adds an initial state */
    void add_initial(State&& decstate){
        this->initial.push_back(decstate);
    };

    /** Adds a sample starting in a decision state */
    void add_sample(const Sample<State,Action>& sample){
        states_from.push_back(sample.state_from());
        actions.push_back(sample.action());
        states_to.push_back(sample.state_to());
        rewards.push_back(sample.reward());
        prec_t cumulative_reward_value = sample.reward();
        if (runs.size() > 0 && *runs.rbegin() == sample.run())
            cumulative_reward_value += *cumulative_rewards.rbegin();
        cumulative_rewards.push_back(cumulative_reward_value);
        weights.push_back(sample.weight());
        steps.push_back(sample.step());
        runs.push_back(sample.run());
    };

    /** Adds a sample starting in a decision state */
    void add_sample(State state_from, Action action,
                    State state_to, prec_t reward, prec_t weight,
                    long step, long run){

        states_from.push_back(move(state_from));
        actions.push_back(move(action));
        states_to.push_back(move(state_to));
        rewards.push_back(reward);
        prec_t cumulative_reward_value = reward;
        if (runs.size() > 0 && *runs.rbegin() == run)
            cumulative_reward_value += *cumulative_rewards.rbegin();
        cumulative_rewards.push_back(cumulative_reward_value);
        weights.push_back(weight);
        steps.push_back(step);
        runs.push_back(run);
    }

    /**
    Computes the discounted mean return over all the samples
    \param discount Discount factor
    */
    prec_t mean_return(prec_t discount){
        prec_t result = 0;
        set<int> runs;

        for(size_t si : indices(*this)){
            auto es = get_sample(si);
            result += es.reward() * pow(discount,es.step());
            runs.insert(es.run());
        }

        result /= runs.size();
        return result;
    };

    /** Number of samples */
    size_t size() const {return states_from.size();};

    /** Access to samples */
    Sample<State,Action> get_sample(long i) const{
        assert(i >=0 && size_t(i) < size());
        return Sample<State,Action>(states_from[i],actions[i],states_to[i],
                rewards[i],weights[i],steps[i],runs[i]);};

    /** Access to samples */
    Sample<State,Action> operator[](long i) const{
        return get_sample(i);
    };

    /** List of initial states */
    const vector<State>& get_initial() const{return initial;};

    const vector<State>& get_states_from() const{return states_from;};
    const vector<Action>& get_actions() const{return actions;};
    const vector<State>& get_states_to() const{return states_to;};
    const vector<prec_t>& get_rewards() const{return rewards;};
    const vector<prec_t>& get_cumulative_rewards() const{return cumulative_rewards;};
    const vector<prec_t>& get_weights() const{return weights;};
    const vector<long>& get_runs() const{return runs;};
    const vector<long>& get_steps() const{return steps;};

protected:

    vector<State> states_from;
    vector<Action> actions;
    vector<State> states_to;
    vector<prec_t> rewards;
    vector<prec_t> cumulative_rewards;
    vector<prec_t> weights;
    vector<long> runs;
    vector<long> steps;

    vector<State> initial;
};

/**
A helper function that constructs a samples object based on the simulator
that is provided to it
*/
template<class Sim, class... U>
Samples<typename Sim::State, typename Sim::Action> make_samples(U&&... u){
    return Samples<typename Sim::State, typename Sim::Action>(forward<U>(u)...);
}

// **********************************************************************
// ****** Discrete simulation specialization ******************
// **********************************************************************


/** Samples in which the states and actions are identified by integers. */
using DiscreteSamples = Samples<long,long>;
/** Integral expectation sample */
using DiscreteSample = Sample<long,long>;

/**
Turns arbitrary samples to discrete ones assuming that actions are
\b state \b independent. That is the actions must have consistent names
across states. This assumption can cause problems
when some samples are missing.

The internally-held discrete samples can be accessed and modified
from the outside. Also, adding more samples will modify the discrete
samples.

See SampleDiscretizerSD for a version in which action names are
dependent on states.

A new hash function can be defined as follows:
\code
namespace std{
    template<> struct hash<pair<int,int>>{
        size_t operator()(pair<int,int> const& s) const{
            boost::hash<pair<int,int>> h;
            return h(s);
        };
    }
};

\endcode

\tparam State Type of state in the source samples
\tparam Action Type of action in the source samples
\tparam Shash Hash function for states
\tparam Ahash Hash function for actions

A hash function hash<type> for each sample type must exists.
*/
template<   typename State,
            typename Action,
            typename SHash = std::hash<State>,
            typename AHash = std::hash<Action>>
class SampleDiscretizerSI{
public:

    /** Constructs new internal discrete samples*/
    SampleDiscretizerSI() : discretesamples(make_shared<DiscreteSamples>()),
        action_map(), state_map() {};

    /** Adds samples to the discrete samples */
    void add_samples(const Samples<State,Action>& samples){

        // initial states
        for(const State& ins : samples.get_initial()){
            discretesamples->add_initial(add_state(ins));
        }

        // samples
        for(auto si : indices(samples)){
            const auto ds = samples.get_sample(si);
            discretesamples->add_sample(
                                     add_state(ds.state_from()),
                                     add_action(ds.action()),
                                     add_state(ds.state_to()),
                                     ds.reward(), ds.weight(),
                                     ds.step(), ds.run());
        }
    }


    /** Returns a state index, and creates a new one if it does not exists */
    long add_state(const State& dstate){
        auto iter = state_map.find(dstate);
        long index;
        if(iter == state_map.end()){
            index = state_map.size();
            state_map[dstate] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a action index, and creates a new one if it does not exists */
    long add_action(const Action& action){
        auto iter = action_map.find(action);
        long index;
        if(iter == action_map.end()){
            index = action_map.size();
            action_map[action] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a shared pointer to the discrete samples */
    shared_ptr<DiscreteSamples> get_discrete(){return discretesamples;};

protected:
    shared_ptr<DiscreteSamples> discretesamples;

    unordered_map<Action,long,AHash> action_map;
    unordered_map<State,long,SHash> state_map;
};


/**
Turns arbitrary samples to discrete ones (with continuous numbers assigned to states)
assuming that actions are \b state \b dependent.

The internally-held discrete samples can be accessed and modified
from the outside. Also, adding more samples will modify the discrete
samples.

See SampleDiscretizerSI for a version in which action names are
independent of states.

A new hash function can be defined as follows:
\code
namespace std{
    template<> struct hash<pair<int,int>>{
        size_t operator()(pair<int,int> const& s) const{
            boost::hash<pair<int,int>> h;
            return h(s);
        };
    }
};
\endcode

\tparam State Type of state in the source samples
\tparam Action Type of action in the source samples
\tparam SAhash Hash function for pair<State, Action>
\tparam Shash Hash function for decision states

A hash function hash<type> for each sample type must exists.
*/
template<
    typename State,
    typename Action,
    typename SAHash = std::hash<pair<State,
                                     Action>>,
    typename SHash = std::hash<State> >
class SampleDiscretizerSD{
public:

    /** Constructs new internal discrete samples*/
    SampleDiscretizerSD() : discretesamples(make_shared<DiscreteSamples>()), action_map(),
                            action_count(), state_map() {};

    /** Adds samples to the discrete samples */
    void add_samples(const Samples<State,Action>& samples){

        // initial states
        for(const auto& ins : samples.get_initial()){
            discretesamples->add_initial(add_state(ins));
        }

        // transition samples
        for(auto si : indices(samples)){

            const auto es = samples.get_sample(si);

            discretesamples->add_sample(add_state(es.state_from()),
                                        add_action(es.state_from(), es.action()),
                                        add_state(es.state_to()),
                                        es.reward(), es.weight(),
                                        es.step(), es.run());
        }
    }

    /** Returns a state index, and creates a new one if it does not exists */
    long add_state(const State& dstate){
        auto iter = state_map.find(dstate);
        long index;
        if(iter == state_map.end()){
            index = state_map.size();
            state_map[dstate] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns an action index, and creates a new one if it does not exists */
    long add_action(const State& dstate, const Action& action){
        auto da = make_pair(dstate, action);
        auto iter = action_map.find(da);
        long index;
        if(iter == action_map.end()){
            index = (action_count[dstate]++);
            action_map[da] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a shared pointer to the discrete samples */
    shared_ptr<DiscreteSamples> get_discrete(){return discretesamples;};

protected:
    shared_ptr<DiscreteSamples> discretesamples;

    unordered_map<pair<State,Action>,long,SAHash> action_map;

    /** keeps the number of actions for each state */
    unordered_map<State,long,SHash> action_count;
    unordered_map<State,long,SHash> state_map;
};



/**
Constructs an MDP from integer samples.

Integer samples: All states and actions are identified by integers.


\a Input: Sample set \f$ \Sigma = (s_i, a_i, s_i', r_i, w_i)_{i=0}^{m-1} \f$ \n
\a Output: An MDP such that:
    - States: \f$ \mathcal{S} = \bigcup_{i=0}^{m-1} \{ s_i \} \cup \bigcup_{i=0}^{m-1} \{ s_i' \} \f$
    - Actions: \f$ \mathcal{A} = \bigcup_{i=0}^{m-1} \{ a_i \} \f$
    - Transition probabilities:
        \f[ P(s,a,s') = \frac{\sum_{i=0}^{m-1} w_i 1\{ s = s_i, a = a_i, s' = s_i' \} }
            { \sum_{i=0}^{m-1} w_i 1\{ s = s_i, a = a_i \} } \f]
    - Rewards:
        \f[ r(s,a,s') = \frac{\sum_{i=0}^{m-1} r_i w_i 1\{ s = s_i, a = a_i, s' = s_i' \} }
            { \sum_{i=0}^{m-1} w_i 1\{ s = s_i, a = a_i, s' = s_i' \} } \f]


The class also tracks cumulative weights of state-action samples \f$ z \f$:
\f[ z(s,a) = \sum_{i=0}^{m-1} w_i 1\{ s = s_i, a = a_i \}  \f]
If \f$ z(s,a) = 0 \f$ then the action \f$ a \f$ is marked as invalid.
There is some extra memory penalty due to storing these weights.

\a Important: Actions that are not sampled (no samples per that state
and action pair) are labeled as invalid and are not included in the computation
of value function or the solution. For example, if there is an action 1 in state zero
but there are no samples that include action 0 then action 0 is still created, but is
ignored when computing the value function.

When sample sets are added by multiple calls of SampledMDP::add_samples, the results is the
same as if all the individual sample sets were combined and added together. See SampledMDP::add_samples
for more details.
*/
class SampledMDP{
public:

    /** Constructs an empty MDP from discrete samples */
    SampledMDP(): mdp(make_shared<MDP>()), initial(), state_action_weights() {}


    /**
    Constructs or adds states and actions based on the
    provided samples.

    Sample sets can be added iteratively. Assume that the current
    transition probabilities are constructed based on a sample set
    \f$ \Sigma = (s_i, a_i, s_i', r_i, w_i)_{i=0}^{m-1} \f$ and add_samples
    is called with sample set \f$ \Sigma' = (s_j, a_j, s_j', r_j, w_j)_{i=m}^{n-1} \f$.
    The result is the same as if simultaneously adding samples \f$ 0 \ldots (n-1) \f$.

    New MDP values are updates as follows:
        - Cumulative state-action weights \f$ z'\f$:
            \f[ z'(s,a) =  z(s,a) + \sum_{j=m}^{n-1} w_j 1\{ s = s_j, a = a_j \} \f]
        - Transition probabilities \f$ P \f$:
            \f{align*}{
            P'(s,a,s') &= \frac{z(s,a) * P(s,a,s') +
                        \sum_{j=m}^{n-1} w_j 1\{ s = s_j, a = a_j, s' = s_j' \} }
                            { z'(s,a) } = \\
                &= \frac{P(s,a,s') +
                (1 / z(s,a)) \sum_{j=m}^{n-1} w_j 1\{ s = s_j, a = a_j, s' = s_j' \} }
                { z'(s,a) / z(s,a)  }

            \f}
            The denominator is computed implicitly by normalizing transition probabilities.
        - Rewards \f$ r' \f$:
            \f{align*}
            r'(s,a,s')
                &= \frac{r(s,a,s') z(s,a) P(s,a,s') +
                       \sum_{j=m}^{n-1} r_j w_j 1\{ s = s_j, a = a_j, s' = s_j' \}}
                       {z'(s,a)P'(s,a,s')} \\
            r'(s,a,s')
                &= \frac{r(s,a,s') z(s,a) P(s,a,s') +
                       \sum_{j=m}^{n-1} r_j w_j 1\{ s = s_j, a = a_j, s' = s_j' \}}
                       {z'(s,a)P'(s,a,s')} \\
                &= \frac{r(s,a,s') P(s,a,s') +
                     \sum_{j=m}^{n-1}r_j (w_j/z(s,a)) 1\{ s = s_j, a = a_j, s' = s_j' \}}
                       {z'(s,a)P'(s,a,s')/ z(s,a)} \\
                &= \frac{r(s,a,s') P(s,a,s') +
                    \sum_{j=m}^{n-1} r_j (w_j/z(s,a) 1\{ s = s_j, a = a_j, s' = s_j' \}}
                       {P(s,a,s') + \sum_{j=m}^{n-1} (w_j/z(s,a)) 1\{ s = s_j, a = a_j, s' = s_j' \}}
            \f}
            The last line follows from the definition of \f$ P(s,a,s') \f$.
            This corresponds to the operation of Transition::add_sample
            repeatedly for \f$ j = m \ldots (n-1) \f$ with
            \f{align*}
            p &= (w_j/z(s,a)) 1\{ s = s_j, a = a_j, s' = s_j' \}\\
            r &= r_j \f}.

    \param samples New sample set to add to transition probabilities and
                    rewards
    */
    void add_samples(const DiscreteSamples& samples){
        // copy the state and action counts to be
        auto old_state_action_weights = state_action_weights;

        // add transition samples
        for(size_t si : indices(samples)){

            DiscreteSample s = samples.get_sample(si);

            // -----------------
            // Computes sample weights:
            // the idea is to normalize new samples by the same
            // value as the existing samples and then re-normalize
            // this is linear complexity
            // -----------------


            // weight used to normalize old data
            prec_t weight = 1.0; // this needs to be initialized to 1.0
            // whether the sample weight has been initialized
            bool weight_initialized = false;

            // resize transition counts
            // the actual values are updated later
            if((size_t) s.state_from() >= state_action_weights.size()){
                state_action_weights.resize(s.state_from()+1);

                // we know that the value will not be found in old data
                weight_initialized = true;
            }

            // check if we have something for the action
            numvec& actioncount = state_action_weights[s.state_from()];
            if((size_t)s.action() >= actioncount.size()){
                actioncount.resize(s.action()+1);

                // we know that the value will not be found in old data
                weight_initialized = true;
            }

            // update the new count
            assert(size_t(s.state_from()) < state_action_weights.size());
            assert(size_t(s.action()) < state_action_weights[s.state_from()].size());

            state_action_weights[s.state_from()][s.action()] += s.weight();

            // get number of existing transitions
            // this is only run when we do not know if we have any prior
            // sample
            if(!weight_initialized &&
                    (size_t(s.state_from()) < old_state_action_weights.size()) &&
                    (size_t(s.action()) < old_state_action_weights[s.state_from()].size())) {

                size_t cnt = old_state_action_weights[s.state_from()][s.action()];

                // adjust the weight of the new sample to be consistent
                // with the previous normalization (use 1.0 if no previous action)
                weight = 1.0 / prec_t(cnt);
            }
            // ---------------------

            // adds a transition
            add_transition( *mdp, s.state_from(), s.action(), s.state_to(),
                            weight*s.weight(),
                            s.reward());
        }


        //  Normalize the transition probabilities and rewards
        mdp->normalize();

        // set initial distribution (not normalized so it is updated correctly when adding more samples
        for(long state : samples.get_initial()){
            if(state > state_count()) throw range_error("Initial state number larger than any transition state.");
            if(state < 0) throw range_error("Initial state with a negative index is invalid.");
            initial.add_sample(state, 1.0, 0.0);
        }
    }

    /** \returns A constant pointer to the internal MDP */
    shared_ptr<const MDP> get_mdp() const {return const_pointer_cast<const MDP>(mdp);}

    /** \returns A modifiable pointer to the internal MDP.
    Take care when changing it. */
    shared_ptr<MDP> get_mdp_mod() {return mdp;}

    /** \returns Initial distribution based on empirical sample data. Could be
     * somewhat expensive because it normalizes the transition. */
    Transition get_initial() const {Transition t = initial; t.normalize(); return t;}

    /** \returns State-action cumulative weights \f$ z \f$.
    See class description for details. */
    vector<vector<prec_t>> get_state_action_weights(){return state_action_weights;}

    /** Returns thenumber of states in the samples (the highest observed index.
    Some may be missing)
    \returns 0 when there are no samples
    */
    long state_count(){return state_action_weights.size();}

protected:

    /** Internal MDP representation */
    shared_ptr<MDP> mdp;

    /** Initial distribution, not normalized */
    Transition initial;

    /** Sample counts */
    vector<vector<prec_t>> state_action_weights;
};


/*
Constructs a robust MDP from integer samples.

In integer samples each decision state, expectation state,
and action are identified by an integer.

There is some extra memory penalty in this class over a plain MDP since it stores
the number of samples observed for each state and action.

Important: Actions that are not sampled (no samples per that state
and action pair) are labeled as invalid and are not included in the computation
of value function or the solution.

*/
//template<typename Model>
//class SampledRMDP{
//public:
//
//    /** Constructs an empty MDP from discrete samples */
//    SampledRMDP();
//
//    /**
//    Constructs or adds states and actions based on the provided samples. Transition
//    probabilities of the existing samples are normalized.
//
//    \param samples Source of the samples
//    */
//    void add_samples(const DiscreteSamples& samples);
//
//    /** \returns A constant pointer to the internal MDP */
//    shared_ptr<const Model> get_rmdp() const {return const_pointer_cast<const Model>(mdp);}
//
//    /** \returns A modifiable pointer to the internal MDP.
//    Take care when changing. */
//    shared_ptr<Model> get_rmdp_mod() {return mdp;}
//
//    /** Initial distribution */
//    Transition get_initial() const {return initial;}
//
//protected:
//
//    /** Internal MDP representation */
//    shared_ptr<Model> mdp;
//
//    /** Initial distribution */
//    Transition initial;
//
//    /** Sample counts */
//    vector<vector<size_t>> state_action_counts;
//};


} // end namespace msen
} // end namespace craam

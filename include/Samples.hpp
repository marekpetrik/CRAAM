#pragma once

#include "definitions.hpp"
#include "RMDP.hpp"

#include <set>
#include <memory>
#include <unordered_map>
#include <functional>
#include <cassert>

#include "cpp11-range-master/range.hpp"

namespace craam{
namespace msen{

using namespace util::lang;
using namespace std;

/// -----------------------------------------------------------------------------------------

/**
Represents the transition between two states.

The footprint of the class could be reduced by removing the number of step and the run.

\tparam State Type defining states
\tparam Action Type defining actions
 */
template <class State, class Action>
class Sample {
public:
    Sample(const State& state_from, const Action& action, const State& state_to,
           prec_t reward, prec_t weight, long step, long run):
        _state_from(state_from), _action(action),
        _state_to(state_to), _reward(reward), _weight(weight), _step(step), _run(run){};

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

/// -----------------------------------------------------------------------------------------

/**
General representation of samples.

\tparam State Type defining states
\tparam Action Type defining actions
 */
template <class State, class Action>
class Samples {
public:

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
        weights.push_back(sample.weight());
        steps.push_back(sample.step());
        runs.push_back(sample.run());
    };

    /** Adds a sample starting in a decision state */
    void add_sample(const State& state_from, const Action& action,
                    const State& state_to, prec_t reward, prec_t weight,
                    long step, long run){

        states_from.push_back(state_from);
        actions.push_back(action);
        states_to.push_back(state_to);
        rewards.push_back(reward);
        weights.push_back(weight);
        steps.push_back(step);
        runs.push_back(run);
    }

    /** Adds a sample starting in a decision state */
    void add_sample(State&& state_from, Action&& action,
                    State&& state_to, prec_t reward, prec_t weight,
                    long step, long run){

        states_from.push_back(state_from);
        actions.push_back(action);
        states_to.push_back(state_to);
        rewards.push_back(reward);
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
        return Sample<State,Action>(states_from[i],actions[i],states_to[i],rewards[i],weights[i],steps[i],runs[i]);};

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
    const vector<prec_t>& get_weights() const{return weights;};
    const vector<long>& get_runs() const{return runs;};
    const vector<long>& get_steps() const{return steps;};

protected:

    vector<State> states_from;
    vector<Action> actions;
    vector<State> states_to;
    vector<prec_t> rewards;
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

/// **********************************************************************
/// ****** Discrete simulation specialization ******************
/// **********************************************************************


/** Samples in which the states and actions are identified by integers. */
using DiscreteSamples = Samples<long,long>;
/** Integral expectation sample */
using DiscreteSample = Sample<long,long>;

/**
Turns arbitrary samples to discrete ones assuming that actions are
*state independent*. That is the actions must have consistent names
across states. This assumption can cause problems
when some samples are missing.

The internally-held discrete samples can be accessed and modified
from the outside. Also, adding more samples will modify the discrete
samples.

See SampleDiscretizerSD for a version in which action names are
dependent on states.

A new hash function can be defined as follows:
namespace std{
    template<> struct hash<pair<int,int>>{
        size_t operator()(pair<int,int> const& s) const{
            boost::hash<pair<int,int>> h;
            return h(s);
        };
    }
};


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
    SampleDiscretizerSI() : discretesamples(make_shared<DiscreteSamples>()){};

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

/// -----------------------------------------------------------------------------------------

/**
Turns arbitrary samples to discrete ones (with continuous numbers assigned to states)
assuming that actions are *state dependent*.

The internally-held discrete samples can be accessed and modified
from the outside. Also, adding more samples will modify the discrete
samples.

See SampleDiscretizerSI for a version in which action names are
independent of states.

A new hash function can be defined as follows:
namespace std{
    template<> struct hash<pair<int,int>>{
        size_t operator()(pair<int,int> const& s) const{
            boost::hash<pair<int,int>> h;
            return h(s);
        };
    }
};

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
    SampleDiscretizerSD() : discretesamples(make_shared<DiscreteSamples>()){};

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


/// -----------------------------------------------------------------------------------------

/**
Constructs MDP from integer samples. In integer samples, each
decision state, expectation state, and action are identified
by an integer.

There is some extra memory penalty in this class since it stores
the number of samples observed for each state and action.

Important: There must be at least one observed sample for each state and action.
Otherwise, the MDP solution will not be defined and the
solver will throw an invalid_argument exception. This can happen when
we have a sample for action 2 but no sample for action 0.
*/
class SampledMDP{
public:

    /** Constructs an empty MDP from discrete samples */
    SampledMDP();

    /**
    Constructs or adds states and actions based on the
    provided samples.

    At this point, the method can be called only once, but
    in the future it should be possible to call it multiple times
    to add more samples.

    \param samples Source of the samples
    */
    void add_samples(const DiscreteSamples& samples);

    /** \returns A constant pointer to the internal MDP */
    shared_ptr<const MDP> get_mdp() const {return const_pointer_cast<const MDP>(mdp);}

    /** Initial distribution */
    Transition get_initial() const {return initial;}

protected:

    /** Internal MDP representation */
    shared_ptr<MDP> mdp;

    /** Initial distribution */
    Transition initial;

    /** Sample counts */
    vector<vector<size_t>> state_action_counts;
};

} // end namespace msen
} // end namespace craam

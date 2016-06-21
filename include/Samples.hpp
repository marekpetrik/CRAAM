#pragma once

#include "definitions.hpp"
#include "RMDP.hpp"

#include <set>
#include <memory>
#include <unordered_map>
#include <functional>

using namespace std;

namespace craam{
namespace msen {

/// -----------------------------------------------------------------------------------------

/**
Represents the transition between two states.

The footprint of the class could be reduced by removing the number of step and the run.

\tparam Sim Simulator class used to generate the sample. Only the members
            defining the types of State and Action are necessary.
 */
template <class Sim>
class Sample {
public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    Sample(const State& state_from, const Action& action, const State& state_to,
           prec_t reward, prec_t weight, long step, long run):
        _state_from(state_from), _action(action),
        _state_to(state_to), _reward(reward), _step(step), _step(step), _run(run){};

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

\tparam Sim Simulator class used to generate the samples. Only the members
            defining the types of State and Action are necessary.
 */
template <typename Sim>
class Samples {
public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    /** Adds an initial state */
    void add_initial(const State& decstate){
        this->initial.push_back(decstate);
    };

    /** Adds a sample starting in a decision state */
    void add_sample(const Sample<Sim>& sample){
        samples.push_back(sample);
    };

    /** Adds a sample starting in a decision state */
    void add_sample(const State& state_from, const Action& action, 
                    const State& state_to, prec_t reward, prec_t weight, 
                    long step, long run){
    
        samples.emplace_back(state_from, action, state_to, reward, weight, 
                             step, run);
    }

    /**
    Computes the discounted mean return over all the samples
    \param discount Discount factor
    */
    prec_t mean_return(prec_t discount){
        prec_t result = 0;
        set<int> runs;

        for(const auto& es : samples){
           result += es.reward * pow(discount,es.step);
           runs.insert(es.run);
        }

        result /= runs.size();
        return result;
    };

    /** List of all samples */
    const vector<Sample<Sim>>& get_samples() const{return samples;};
    /** List of initial states */
    const vector<State>& get_initial() const{return initial;};

protected:

    vector<Sample<Sim>> samples;
    vector<State> initial;

};

/// -----------------------------------------------------------------------------------------


/** Class used to define discrete samples */
class DiscreteSimulator {
public:
    typedef long State;
    typedef long Action;
};

/** Samples in which the states and actions are identified by integers. */
using DiscreteSamples = Samples<DiscreteSimulator>;
/** Integral expectation sample */
using DiscreteSample = Sample<DiscreteSimulator>;


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


\tparam Simulator definition for which samples to use
\tparam Shash Hash function for states
\tparam Ahash Hash function for actions

A hash function hash<type> for each sample type must exists.
*/
template<   typename Sim,
            typename SHash = std::hash<typename Sim::State>,
            typename AHash = std::hash<typename Sim::Action>>
class SampleDiscretizerSI{
public:
    using State = typename Sim::State;

    /** Constructs new internal discrete samples*/
    SampleDiscretizerSI() : discretesamples(make_shared<DiscreteSamples>()){};

    /** Adds samples to the discrete samples */
    void add_samples(const Samples<Sim>& samples){

        // initial states
        for(const State& ins : samples.get_initial()){
            discretesamples->add_initial(add_state(ins));
        }

        // samples
        for(const auto& ds : samples.get_samples()){
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
    long add_action(const typename Sim::Action& action){
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

    unordered_map<typename Sim::Action,long,AHash> action_map;
    unordered_map<typename Sim::State,long,SHash> state_map;
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

\tparam Simulator definition for which samples to use
\tparam SAhash Hash function for pair<State, Action>
\tparam Shash Hash function for decision states

A hash function hash<type> for each sample type must exists.
*/
template<
    typename Sim,
    typename SAHash = std::hash<pair<typename Sim::State,
                                     typename Sim::Action>>,
    typename SHash = std::hash<typename Sim::State> >
class SampleDiscretizerSD{
public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    /** Constructs new internal discrete samples*/
    SampleDiscretizerSD() : discretesamples(make_shared<DiscreteSamples>()){};

    /** Adds samples to the discrete samples */
    void add_samples(const Samples<Sim>& samples){

        // initial states
        for(const auto& ins : samples.get_initial()){
            discretesamples->add_initial(add_state(ins));
        }

        // transition samples
        for(const auto& es : samples.get_samples()){
            discretesamples->add_sample(add_state(es.state_from()),
                                        add_action(es.state_from(), es.action()),
                                        add_state(es.state_to()),
                                        es.reward(), es.weight(),
                                        es.step, es.run);
        }
    }

    /** Returns a state index, and creates a new one if it does not exists */
    long add_state(const typename Sim::State& dstate){
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

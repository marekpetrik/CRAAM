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

/**
Represents the transition from an expectation state to a
a decision state.

\tparam Sim Simulator class used to generate the sample. Only the members
            defining the types of DState, Action, and EState are necessary.
*/
template <class Sim>
struct ESample {

    const typename Sim::EState expstate_from;
    const typename Sim::DState decstate_to;
    const prec_t reward;
    const prec_t weight;
    const long step;
    const long run;

    ESample(const typename Sim::EState& expstate_from, const typename Sim::DState& decstate_to,
              prec_t reward, prec_t weight, long step, long run):
                    expstate_from(expstate_from), decstate_to(decstate_to),
                    reward(reward), weight(weight), step(step), run(run){};
};

/**
Represents the transition from a decision state to an expectation state.

\tparam Sim Simulator class used to generate the sample. Only the members
            defining the types of DState, Action, and EState are necessary.
 */
template <class Sim>
struct DSample {

    const typename Sim::DState decstate_from;
    const typename Sim::Action action;
    const typename Sim::EState expstate_to;
    const long step;
    const long run;

    DSample(const typename Sim::DState& decstate_from, const typename Sim::Action& action,
             const typename Sim::EState& expstate_to, long step, long run):
                decstate_from(decstate_from), action(action),
                expstate_to(expstate_to), step(step), run(run){};
};

/**
General representation of samples.

\tparam Sim Simulator class used to generate the samples. Only the members
            defining the types of DState, Action, and EState are necessary.
 */
template <typename Sim>
class Samples {
public:
    vector<DSample<Sim>> decsamples;
    vector<typename Sim::DState> initial;
    vector<ESample<Sim>> expsamples;

public:

    /** Adds an initial state */
    void add_initial(const typename Sim::DState& decstate){
        this->initial.push_back(decstate);
    };

    /** Adds a sample starting in a decision state */
    void add_dec(const DSample<Sim>& decsample){
        this->decsamples.push_back(decsample);
    };

    /** Adds a sample starting in an expectation state */
    void add_exp(const ESample<Sim>& expsample){
        this->expsamples.push_back(expsample);
    };

    /**
    Computes the discounted mean return over all the samples
    \param discount Discount factor
    */
    prec_t mean_return(prec_t discount){
        prec_t result = 0;
        set<int> runs;

        for(const auto& es : expsamples){
           result += es.reward * pow(discount,es.step);
           runs.insert(es.run);
        }

        result /= runs.size();
        return result;
    };
};

// define discrete

/** Class used to define discrete samples */
class DiscreteSimulator {
public:
    typedef long DState;
    typedef long Action;
    typedef long EState;
};

/** Samples in which the states and actions are identified by integers. */
typedef Samples<DiscreteSimulator> DiscreteSamples;
/** Integral expectation sample */
typedef ESample<DiscreteSimulator> DiscreteESample;
/** Integral decision sample */
typedef DSample<DiscreteSimulator> DiscreteDSample;

/**
Turns arbitrary samples to discrete ones assuming that actions are
*state independent*. That is the actions must have consistent names
across states. This assumption can cause problems
when some samples are missing.

The internally-held discrete samples can be accessed and modified
from the outside. Also, adding more samples will modify the discrete
samples.

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
\tparam Dhash Hash function for decision states
\tparam Ahash Hash function for actions
\tparam Ehash Hash function for expectation states

A hash function hash<type> for each sample type must exists.
*/
template<
    typename Sim,
    typename DHash = std::hash<typename Sim::DState>,
    typename AHash = std::hash<typename Sim::Action>,
    typename EHash = std::hash<typename Sim::EState>
    >
class SampleDiscretizerSI{
public:
    /** Constructs new internal discrete samples*/
    SampleDiscretizerSI() : discretesamples(make_shared<DiscreteSamples>()){};

    /** Returns a decision state index, and creates a new one if it does not exists */
    long add_dstate(const typename Sim::DState& dstate){
        auto iter = dstate_map.find(dstate);
        long index;
        if(iter == dstate_map.end()){
            index = dstate_map.size();
            dstate_map[dstate] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a expectation state index, and creates a new one if it does not exists */
    long add_estate(const typename Sim::EState& estate){
        auto iter = estate_map.find(estate);
        long index;
        if(iter == estate_map.end()){
            index = estate_map.size();
            estate_map[estate] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a expectation state index, and creates a new one if it does not exists */
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

    /** Adds samples to the discrete samples */
    void add_samples(const Samples<Sim>& samples){

        // initial states
        for(const auto& ins : samples.initial){
            discretesamples->add_initial(add_dstate(ins));
        }

        // decision samples
        for(const auto& ds : samples.decsamples){
            discretesamples->add_dec(DiscreteDSample(
                                     add_dstate(ds.decstate_from),
                                     add_action(ds.action),
                                     add_estate(ds.expstate_to),
                                     ds.step, ds.run));
        }

        // expectation samples
        for(const auto& es : samples.expsamples){
            discretesamples->add_exp(DiscreteESample(
                                     add_estate(es.expstate_from),
                                     add_dstate(es.decstate_to),
                                     es.reward, es.weight,
                                     es.step, es.run));
        }
    }

    /** Returns a shared pointer to the discrete samples */
    shared_ptr<DiscreteSamples> get_discrete(){return discretesamples;};

protected:
    shared_ptr<DiscreteSamples> discretesamples;

    unordered_map<typename Sim::Action,long,AHash> action_map;
    unordered_map<typename Sim::DState,long,DHash> dstate_map;
    unordered_map<typename Sim::EState,long,EHash> estate_map;
};


/**
Turns arbitrary samples to discrete ones assuming that actions are
*state dependent*.

The internally-held discrete samples can be accessed and modified
from the outside. Also, adding more samples will modify the discrete
samples.

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
\tparam DAhash Hash function for pair<DState, Action>
\tparam Dhash Hash function for decision states
\tparam Ehash Hash function for expectation states

A hash function hash<type> for each sample type must exists.
*/
template<
    typename Sim,
    typename DAHash = std::hash<pair<typename Sim::DState,
                                     typename Sim::Action>>,
    typename DHash = std::hash<typename Sim::DState>,
    typename EHash = std::hash<typename Sim::EState>
    >
class SampleDiscretizerSD{
public:
    /** Constructs new internal discrete samples*/
    SampleDiscretizerSD() : discretesamples(make_shared<DiscreteSamples>()){};

    /** Returns a decision state index, and creates a new one if it does not exists */
    long add_dstate(const typename Sim::DState& dstate){
        auto iter = dstate_map.find(dstate);
        long index;
        if(iter == dstate_map.end()){
            index = dstate_map.size();
            dstate_map[dstate] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a expectation state index, and creates a new one if it does not exists */
    long add_estate(const typename Sim::EState& estate){
        auto iter = estate_map.find(estate);
        long index;
        if(iter == estate_map.end()){
            index = estate_map.size();
            estate_map[estate] = index;
        }
        else{
            index = iter->second;
        }
        return index;
    }

    /** Returns a expectation state index, and creates a new one if it does not exists */
    long add_action(const typename Sim::DState& dstate, const typename Sim::Action& action){
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

    /** Adds samples to the discrete samples */
    void add_samples(const Samples<Sim>& samples){

        // initial states
        for(const auto& ins : samples.initial){
            discretesamples->add_initial(add_dstate(ins));
        }

        // decision samples
        for(const auto& ds : samples.decsamples){
            discretesamples->add_dec(DiscreteDSample(
                                     add_dstate(ds.decstate_from),
                                     add_action(ds.decstate_from, ds.action),
                                     add_estate(ds.expstate_to),
                                     ds.step, ds.run));
        }

        // expectation samples
        for(const auto& es : samples.expsamples){
            discretesamples->add_exp(DiscreteESample(
                                     add_estate(es.expstate_from),
                                     add_dstate(es.decstate_to),
                                     es.reward, es.weight,
                                     es.step, es.run));
        }
    }

    /** Returns a shared pointer to the discrete samples */
    shared_ptr<DiscreteSamples> get_discrete(){return discretesamples;};

protected:
    shared_ptr<DiscreteSamples> discretesamples;

    unordered_map<pair<typename Sim::DState,
                       typename Sim::Action>,
                  long,DAHash> action_map;

    /** keeps the number of actions for each state */
    unordered_map<typename Sim::DState,long,DHash> action_count;

    unordered_map<typename Sim::DState,long,DHash> dstate_map;
    unordered_map<typename Sim::EState,long,EHash> estate_map;
};


/**
Constructs MDP from integer samples. In integer samples, each
decision state, expectation state, and action are identified
by an integer.

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

    /** Whether it has been initialized */
    bool initialized = false;
};

/**
Constructs MDP from integer samples. This is similar to SampledMDP, but
there are separate states for sampled decision and expectation states.
This approach also requires adjusting the discount factor and additional
functions mapping value function from one representation to the other.

The main advantage of this approach is that it can reduce the computational complexity
when there are transitions from multiple decision states to a single expectation
state.
*/
class SampledMDP_Exp{
public:
    // TODO: copy the Python code

};

} // end namespace msen
} // end namespace craam

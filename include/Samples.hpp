#pragma once

#include <set>

#include "definitions.hpp"

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
        reward(reward), weight(weight), step(step), run(run)
                  {};
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
        expstate_to(expstate_to), step(step), run(run)
        {};
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
    /**
    Adds a sample starting in a decision state
    */
    void add_dec(const DSample<Sim>& decsample){
        this->decsamples.push_back(decsample);
    };

    /**
    Adds an initial state
    */
    void add_initial(const typename Sim::DState& decstate){
        this->initial.push_back(decstate);
    };

    /** Adds a sample starting in an expectation state */
    void add_exp(const ESample<Sim>& expsample){
        this->expsamples.push_back(expsample);
    };

    /**
    Computes the discounted mean return over all the
    samples
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


/** Class used to define discrete samples */
class DiscreteSimulator {
    typedef long DState;
    typedef long Action;
    typedef long EState;
};

/** Samples in which the states and actions are identified by integers. */
typedef Samples<DiscreteSimulator> DiscreteSamples;

/**
Constructs MDP from integer samples. In integer samples, each
decision state, expectation state, and action are identified
by an integer,
*/
class SampledMDP{
public:


};

/**
Constructs MDP from integer samples. This is similar to SampledMDP, but
there are separate states for sampled decision and expectation states.
This approach also requires adjusting the discount factor and additional
functions mapping value function from one representation to the other.

The main advantage of this approach is a possible dramatic simplification
when there are transitions from multiple decision states to a single expectation
state.
*/
class SampledMDP_Exp{

    // TODO: copy the Python code

};

} // end namespace msen
} // end namespace craam

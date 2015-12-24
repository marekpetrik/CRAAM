#pragma once

#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>


#include "Samples.hpp"
#include "definitions.hpp"


using namespace std;

namespace craam{
namespace msen {


/**
A random policy with state-dependent action sets.

\tparam Sim Simulator class for which the policy is to be constructed.
            Must implement an instance method actions(DState).
 */
template<class Sim>
class RandomPolicySD {

public:

    RandomPolicySD(const Sim& sim, random_device::result_type seed = random_device{}()) : sim(sim), gen(seed)
    {};

    /** Returns a random action */
    typename Sim::Action operator() (typename Sim::DState dstate){
        const vector<typename Sim::Action>&& actions = sim.actions(dstate);

        auto actioncount = actions.size();
        uniform_int_distribution<> dst(0,actioncount-1);

        return actions[dst(gen)];
    };

private:
    const Sim& sim;
    default_random_engine gen;

};

/**
An object that behaves as a random policy for problems
with state-independent action sets.

The actions are copied internally.

\tparam Sim Simulator class for which the policy is to be constructed.
            Must implement an instance method actions().
 */
template<class Sim>
class RandomPolicySI {
public:

    RandomPolicySI(const Sim& sim, random_device::result_type seed = random_device{}())
        : sim(sim), gen(seed), actions(sim.actions())
    {};

    /** Returns a random action */
    typename Sim::Action operator() (typename Sim::DState dstate){
        auto actioncount = actions.size();
        uniform_int_distribution<> dst(0,actioncount-1);

        return actions[dst(gen)];
    };

private:

    const Sim& sim;
    default_random_engine gen;
    const vector<typename Sim::Action> actions;   // list of actions is constant
};

//-----------------------------------------------------------------------------------

/**
Runs the simulator and generates samples.

This method assumes that the simulator can state simulation in any state. There may be
an internal state, however, which is independent of the transitions; for example this may be
the internal state of the random number generator.

States and actions are passed by value everywhere and therefore it is important that
they are lightweight objects.

An example definition of a simulator should have the following methods:
\code
/// This class represents a stateless simular, but the non-constant
/// functions may change the state of the random number generator
class Simulator{
public:
    /// Type of decision states
    typedef dec_state_type DState;
    /// Type of actions
    typedef action_type Action;
    /// Type of expectation states
    typedef exp_state_type EState;

    /// Returns a sample from the initial states.
    DState init_state();
    /// Returns an expectation state that follows a decision state and an action
    EState transition_dec(DState, Action);
    /// Returns a sample of the reward and a decision state following an expectation state
    pair<double,DState> transition_exp(EState);
    /// Checks whether the decision state is terminal
    bool end_condition(DState) const;

    /// ** The following functions are not necessary for the simulation
    /// State dependent action list (use RandomPolicySD)
    vector<Action> actions(DState)  const;
    /// State-indpendent action list (use RandomPolicySI)
    vector<Action> actions const;
}
\endcode

\tparam Sim Simulator class used in the simulation. See the main description for the methods
            the simulator must provide.
\tparam SampleType Class used to hold the samples.

\param sim Simulator that holds the properties needed by the simulator
\param samples Add the result of the simulation to this object
\param policy Policy function
\param horizon Number of steps
\param prob_term The probability of termination in each step

 */
template<class Sim,class SampleType=Samples<Sim>>
void simulate_stateless(
            Sim& sim, SampleType& samples,
            const function<typename Sim::Action(typename Sim::DState&)>& policy,
            long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
            random_device::result_type seed = random_device{}()){

    long transitions = 0;

    // initialize random numbers when appropriate
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    for(auto run=0l; run < runs; run++){

        typename Sim::DState&& decstate = sim.init_state();
        samples.add_initial(decstate);

        for(auto step=0l; step < horizon; step++){
            if(sim.end_condition(decstate))
                break;
            if(tran_limit > 0 && transitions > tran_limit)
                break;

            typename Sim::Action&& action = policy(decstate);
            typename Sim::EState&& expstate = sim.transition_dec(decstate,action);

            samples.add_dec(DSample<Sim> (decstate, action, expstate, step, run));

            auto&& rewardstate = sim.transition_exp(expstate);

            auto reward = rewardstate.first;
            decstate = rewardstate.second;

            samples.add_exp(ESample<Sim>(expstate, decstate, reward, 1.0, step, run));

            // test the termination probability only after at least one transition
            if(prob_term > 0.0){
                if( distribution(generator) <= prob_term)
                    break;
            }
            transitions++;
        };

        if(tran_limit > 0 && transitions > tran_limit)
            break;
    }
}

/**
Overloaded function that creates the samples object
\return Samples
*/
template<class Sim,class SampleType=Samples<Sim>>
unique_ptr<SampleType> simulate_stateless(
            Sim& sim,
            const function<typename Sim::Action(typename Sim::DState&)>& policy,
            long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
            random_device::result_type seed = random_device{}()){

    auto samples = make_unique<SampleType>();
    simulate_stateless<Sim,SampleType>(sim, *samples, policy, horizon, runs,
                                        tran_limit, prob_term, seed);
    return samples;
}

} // end namespace msen
} // end namespace craam

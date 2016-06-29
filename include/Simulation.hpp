#pragma once

#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>


#include "Samples.hpp"
#include "definitions.hpp"
#include "cpp11-range-master/range.hpp"


namespace craam{
namespace msen {


using namespace std;
using namespace util::lang;

///-----------------------------------------------------------------------------------

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
    /// Type of states
    typedef state_type State;
    
    /// Type of actions
    typedef action_type Action;

    /// Returns a sample from the initial states.
    State init_state();
    
    /// Returns a sample of the reward and a decision state following an expectation state
    pair<double,State> transition(State, Action);

    /// Checks whether the decision state is terminal
    bool end_condition(State) const;

    /// ** The following functions are not necessary for the simulation
    /// State dependent actions, with discrete number of actions (long id each action)
    /// use -1 if actions are not discrete
    long action_count(State) const;

    /// State dependent action with the given index 
    Action action(State, index) const;
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
template<class Sim, class SampleType=Samples<Sim>>
void simulate(
            Sim& sim, SampleType& samples,
            const function<typename Sim::Action(typename Sim::State&)>& policy,
            long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
            random_device::result_type seed = random_device{}()){

    long transitions = 0;

    // initialize random numbers when appropriate
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    for(auto run=0l; run < runs; run++){

        typename Sim::State&& state = sim.init_state();
        samples.add_initial(state);

        for(auto step : range(0l,horizon)){
            // check form termination conditions
            if(sim.end_condition(state) || (tran_limit > 0 && transitions > tran_limit) )
                break;

            auto&& action = policy(state);
            auto&& rewardState = sim.transition(state,action);

            auto reward = rewardState.first;
            auto nextstate = move(rewardState.second);

            samples.add_sample(move(state), move(action), nextstate, reward, 1.0, step, run);
            state = move(nextstate);

            // test the termination probability only after at least one transition
            if( (prob_term > 0.0) && (distribution(generator) <= prob_term) )
                break;
            transitions++;
        };

        if(tran_limit > 0 && transitions > tran_limit)
            break;
    }
}

/**
Runs the simulator and generates samples.

See the other version of the method for more details. This variant
constructs and returns the samples object.
*/
template<class Sim, class SampleType=Samples<Sim>>
SampleType simulate(
            Sim& sim,
            const function<typename Sim::Action(typename Sim::State&)>& policy,
            long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
            random_device::result_type seed = random_device{}()){

    SampleType samples = SampleType();
    simulate(sim, samples, policy, horizon, runs, tran_limit, prob_term, seed);
    return samples;
}

/// ************************************************************************************
/// **** Random(ized) policies ****
/// ************************************************************************************

/**
A random policy with state-dependent action sets which are discrete.

\tparam Sim Simulator class for which the policy is to be constructed.
            Must implement an instance method actions(State).
 */
template<class Sim>
class RandomPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;


    RandomPolicy(const Sim& sim, random_device::result_type seed = random_device{}()) : sim(sim), gen(seed)
    {};

    /** Returns a random action */
    Action operator() (State state){
        uniform_int_distribution<> dst(0,sim.action_count(state)-1);
        return sim.action(state,dst(gen));
    };

private:
    const Sim& sim;
    default_random_engine gen;
};



/// ************************************************************************************
/// **** MDP simulation ****
/// ************************************************************************************

/**
A simulator that behaves as the provided MDP. A state of MDP.size() is
considered to be the terminal state.

If the sum of all transitions is less than 1, then the remainder is assumed to 
be the probability of transitioning to the terminal state.
*/
class ModelSimulator{

public:
    /// Type of states
    typedef long State;
    /// Type of actions
    typedef long Action;

    /** Build a model simulator and share and MDP */
    ModelSimulator(shared_ptr<const MDP> mdp, const Transition& initial):
        mdp(mdp), initial(initial){};

    /** Build a model simulator and share and MDP */
    ModelSimulator(shared_ptr<MDP> mdp, const Transition& initial) : 
        ModelSimulator(const_pointer_cast<const MDP>(mdp), initial) {};

    /// Returns a sample from the initial states.
    State init_state();
    
    /// Returns a sample of the reward and a decision state following an expectation state
    pair<double,State> transition(State, Action);

    /// Checks whether the decision state is terminal
    bool end_condition(State s) const 
        {return size_t(s) >= mdp->size();};

    /// State dependent action list 
    size_t action_count(State state) const 
        {return (*mdp)[state].size();};
    
    /// Returns an action with the given index
    Action action(State, long index) const
        {return index;};

protected:
    /** MDP used for the simulation */
    shared_ptr<const MDP> mdp;
    /** Initial distribution */
    Transition initial;
};


} // end namespace msen
} // end namespace craam

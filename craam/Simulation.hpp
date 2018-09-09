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

#include "Samples.hpp"
#include "definitions.hpp"

#include <rm/range.hpp>
#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>
#include <algorithm>
#include <cmath>
#include <string>



namespace craam{
namespace msen {

using namespace std;
using namespace util::lang;

///-----------------------------------------------------------------------------------

/**
Runs the simulator and generates samples.

This method assumes that the simulator can start simulation in any state. There may be
an internal state, however, which is independent of the transitions; for example this may be
the internal state of the random number generator.

States and actions are passed by value everywhere (moved when appropriate) and 
therefore it is important that they are lightweight objects.

A simulator should have the following methods:
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

    /// ** The following functions are not necessary for simulation
    /// ** but are used to generate policies (random(ized) )

    /// State dependent actions, with discrete number of actions (long id each action)
    /// use -1 if infinite number of actions are available
    long action_count(State) const;

    /// State dependent action with the given index 
    Action action(State, index) const;
}
\endcode

\tparam Sim Simulator class used in the simulation. See the main description for the methods that
            the simulator must provide.
\tparam SampleType Class used to hold the samples.

\param sim Simulator that holds the properties needed by the simulator
\param samples Add the result of the simulation to this object
\param policy Policy function
\param horizon Number of steps
\param prob_term The probability of termination in each step
 */
template<class Sim, class SampleType=Samples<typename Sim::State, typename Sim::Action>>
void simulate(
            Sim& sim, SampleType& samples,
            const function<typename Sim::Action(typename Sim::State&)>& policy,
            long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
            random_device::result_type seed = random_device{}()){

    long transitions = 0;

    // initialize random numbers to be used with random termination
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    for(auto run=0l; run < runs; run++){

        typename Sim::State state = sim.init_state();
        samples.add_initial(state);

        for(auto step : range(0l,horizon)){
            // check form termination conditions
            if(sim.end_condition(state) || (tran_limit > 0 && transitions > tran_limit) )
                break;

            auto action = policy(state);
            auto reward_state = sim.transition(state,action);

            auto reward = reward_state.first;
            auto nextstate = move(reward_state.second);

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

See the overloaded version of the method for more details. This variant
constructs and returns the samples object.

\returns Set of samples
*/
template<class Sim, class SampleType=Samples<typename Sim::State, typename Sim::Action>>
SampleType simulate(
            Sim& sim,
            const function<typename Sim::Action(typename Sim::State&)>& policy,
            long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
            random_device::result_type seed = random_device{}()){

    SampleType samples = SampleType();
    simulate(sim, samples, policy, horizon, runs, tran_limit, prob_term, seed);
    return samples;
}


/**
Runs the simulator and computer the returns from the simulation.

This method assumes that the simulator can start simulation in any state. There may be
an internal state, however, which is independent of the transitions; for example this may be
the internal state of the random number generator.

States and actions are passed by value everywhere (moved when appropriate) and therefore it is important that
they are lightweight objects.


\tparam Sim Simulator class used in the simulation. See the main description for the methods that
            the simulator must provide.

\param sim Simulator that holds the properties needed by the simulator
\param discount Discount to use in the computation
\param policy Policy function
\param horizon Number of steps
\param prob_term The probability of termination in each step

\returns Pair of (states, cumulative returns starting in states)
 */

template<class Sim>
pair<vector<typename Sim::State>, numvec> 
simulate_return(Sim& sim, prec_t discount,
                const function<typename Sim::Action(typename Sim::State&)>& policy,
                long horizon, long runs, prec_t prob_term=0.0,
                random_device::result_type seed = random_device{}()){

    long transitions = 0;
    // initialize random numbers to be used with random termination
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    // pre-initialize output values
    vector<typename Sim::State> start_states(runs);
    numvec returns(runs);

    for(auto run : range(0l,runs)){
        typename Sim::State state = sim.init_state();
        start_states[run] = state;

        prec_t runreturn = 0;
        for(auto step : range(0l,horizon)){
            // check from-state termination conditions
            if(sim.end_condition(state))
                break;

            auto action = policy(state);
            auto reward_state = sim.transition(state,action);
            
            auto reward = reward_state.first;
            auto nextstate = move(reward_state.second);

            runreturn += reward * pow(discount, step);
            state = move(nextstate);
            // test the termination probability only after at least one transition
            if( (prob_term > 0.0) && (distribution(generator) <= prob_term) )
                break;
            transitions++;
        };
        returns[run] = runreturn;
    }
    return make_pair(move(start_states), move(returns));
}

// ************************************************************************************
// **** Random(ized) policies ****
// ************************************************************************************

/**
A random policy with state-dependent action sets which are discrete.

Important: Retains the reference to the simulator from which it comes. If the
simulator is destroyed in the meantime, the behavior is not defined.

\tparam Sim Simulator class for which the policy is to be constructed.
            Must implement an instance method actions(State).
 */
template<class Sim>
class RandomPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    RandomPolicy(const Sim& sim, random_device::result_type seed = random_device{}()) : 
                sim(sim), gen(seed){};

    /** Returns a random action */
    Action operator() (State state){
        vector<Action> valid_actions = sim.get_valid_actions(state);
        uniform_int_distribution<long> dst(0,valid_actions.size()-1);
        return valid_actions[dst(gen)];
    };

private:
    /// Internal reference to the originating simulator
    const Sim& sim;
    /// Random number engine
    default_random_engine gen;
};

/**
A randomized policy that chooses actions according to the provided
vector of probabilities. 

Action probabilities must sum to one for each state. 

State must be convertible to a long index; that is must support
    (explicit) operator long
Actions also have to be indexed. See the definition of simulate.
*/
template<typename Sim>
class RandomizedPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    /**
    Initializes randomized polices, transition probabilities
    for each state. The policy is applicable only to simulators that
    have:

      1) At most as many states as probabilities.size()
      2) At least as many actions are max(probabilities[i].size() | i) 


    \param sim Simulator used with the policy. The reference is retained,
                the object should not be deleted
    \param probabilities List of action probabilities for each state
    */
    RandomizedPolicy(const Sim& sim, const vector<numvec>& probabilities,random_device::result_type seed = random_device{}()):
        gen(seed), distributions(probabilities.size()), sim(sim){

        for(auto pi : indices(probabilities)){
            
            // check that this distribution is correct
            const numvec& prob = probabilities[pi];
            prec_t sum = accumulate(prob.begin(), prob.end(), 0.0);
            
            if(abs(sum - 1) > SOLPREC){
                throw invalid_argument("Action probabilities must sum to 1 in state " + to_string(pi));
            } 
            distributions[pi] = discrete_distribution<long>(prob.begin(), prob.end());
        }
    };

    /** Returns a random action */
    Action operator() (State state){
        // check that the state is valid for this policy
        long sl = static_cast<long>(state);
        assert(sl >= 0 && size_t(sl) < distributions.size());

        auto& dst = distributions[sl];
        // existence of the action is check by the simulator
        return sim.action(state,dst(gen));
    };

protected:

    /// Random number engine
    default_random_engine gen;

    /// List of discrete distributions for all states
    vector<discrete_distribution<long>> distributions;

    /// simulator reference
    const Sim& sim;
};


/**
A deterministic policy that chooses actions according to the provided action index. 

State must be convertible to a long index; that is must support
    (explicit) operator long
Actions also have to be indexed. See the definition of simulate.

*/
template<typename Sim>
class DeterministicPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    /**
    Initializes randomized polices, transition probabilities
    for each state. 

    \param sim Simulator used with the policy. The reference is retained,
                the object should not be deleted
    \param actions Index of action to take for each state 
    */
    DeterministicPolicy(const Sim& sim, indvec actions):
        actions(actions), sim(sim) {};

    /** Returns a random action */
    Action operator() (State state){
        // check that the state is valid for this policy
        long sl = static_cast<long>(state);

        assert(sl >= 0 && size_t(sl) < actions.size());

        // existence of the action is checked by the simulator
        return sim.action(state,actions[sl]);
    };

protected:
    /// List of which action to take in which state
    indvec actions;

    /// simulator reference
    const Sim& sim;
};

/**
A stochastic policy that chooses actions according to the state and the action selection probability.

State must be convertible to a long index; that is must support
    (explicit) operator long

*/
template<typename Sim>
class StochasticPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    /**
    Initializes the action/state probability map

    \param sim Simulator used with the policy. The reference is retained,
                the object should not be deleted
    \param actions Index of action to take for each state
    */
    StochasticPolicy(const Sim& sim, prob_matrix_t actions, random_device::result_type seed = random_device{}()):
        gen(seed), distribution(0,1), actions(actions), sim(sim) {};

    /** Returns an action based on the probability of it being selected from the given state*/
    Action operator() (State state){
        // check that the state is valid for this policy
        long sl = static_cast<long>(state);

        assert(sl >= 0 && size_t(sl) < actions.size());

        // Get the next action based on the change to select that action
        prob_list_t &action_probabilities = actions[sl];
        double selected_probability = distribution(gen);
        double current_probability = 0;
        int action_index = 0;
        for ( prob_list_t::iterator ap_iter = action_probabilities.begin(); ap_iter != action_probabilities.end(); ap_iter++ )
        {
            current_probability += *ap_iter;
            if ( current_probability >= selected_probability )
                return sim.action(state,action_index);
            action_index++;
        }

        //If we got here, something went wrong
        assert(false);
        throw 1;
    };

protected:
    /// Random number engine
    default_random_engine gen;
    uniform_real_distribution<double> distribution;

    /// List of which action to take in which state
    prob_matrix_t actions;

    /// simulator reference
    const Sim& sim;
};

// ************************************************************************************
// **** MDP simulation ****
// ************************************************************************************

/**
A simulator that behaves as the provided MDP. A state of MDP.size() is
considered to be the terminal state.

If the sum of all transitions is less than 1, then the remainder is assumed to 
be the probability of transitioning to the terminal state. 

Any state with an index higher or equal to the number of states is considered to be terminal.
*/
class ModelSimulator{

public:
    /// Type of states
    typedef long State;
    /// Type of actions
    typedef long Action;

    /** 
    Build a model simulator and share and MDP 
    
    The initial transition is copied internally to the object,
    while the MDP object is stored internally.
    */
    ModelSimulator(const shared_ptr<const MDP>& mdp, const Transition& initial, 
                        random_device::result_type seed = random_device{}()) :
                gen(seed), mdp(mdp), initial(initial){

        if(abs(initial.sum_probabilities() - 1) > SOLPREC)
            throw invalid_argument("Initial transition probabilities must sum to 1");
    }

    /** 
    Build a model simulator and share and MDP 
    
    The initial transition is copied internally to the object,
    while the MDP object is stored internally.
    */
    ModelSimulator(const shared_ptr<MDP>& mdp, const Transition& initial,random_device::result_type seed = random_device{}()) : 
        ModelSimulator(const_pointer_cast<const MDP>(mdp), initial, seed) {};

    /// Returns a sample from the initial states.
    State init_state(){
        const numvec& probs = initial.get_probabilities();
        const indvec& inds = initial.get_indices();
        auto dst = discrete_distribution<long>(probs.begin(), probs.end());
        return inds[dst(gen)];
    }
    
    /** 
    Returns a sample of the reward and a decision state following a state

    If the transition probabilities do not sum to 1, then them remainder
    is considered as a probability of transitioning to a terminal state
    (one with an index that is too large; see ModelSimulator::end_condition)

    \param state Current state
    \param action Current action
    */
    pair<double,State> transition(State state, Action action){

        assert(state >= 0 && size_t(state) < mdp->size());
        const auto& mdpstate = (*mdp)[state];

        assert(action >= 0 && size_t(action) < mdpstate.size());
        const auto& mdpaction = mdpstate[action];

        if(!mdpstate.is_valid(action))
            throw invalid_argument("Cannot transition using an invalid action");

        const auto& tran = mdpaction.get_outcome();

        const numvec& probs = tran.get_probabilities();
        const numvec& rews = tran.get_rewards();
        const indvec& inds = tran.get_indices();

        // check if the transition sums to 1, if not use the remainder 
        // as a probability of terminating
        prec_t prob_termination = 1 - tran.sum_probabilities();

        discrete_distribution<long> dst;

        if(prob_termination > SOLPREC){
            // copy the probabilities (there should be a faster way too)
            numvec copy_probs(probs);
            copy_probs.push_back(prob_termination);

            dst = discrete_distribution<long>(copy_probs.begin(), copy_probs.end());
        }else{
            dst = discrete_distribution<long>(probs.begin(), probs.end());
        }

        const size_t nextindex = dst(gen);

        // check if need to transition to a terminal state
        const State nextstate = nextindex < inds.size() ? 
                                inds[nextindex] : mdp->size();

        // reward is zero when transitioning to a terminal state
        const prec_t reward = nextindex < inds.size() ? 
                                rews[nextindex] : 0.0;

        return make_pair(reward, nextstate);
    }

    /**
    Checks whether the decision state is terminal. A state is 
    assumed to be terminal when:
        1. Its index is too large. That is, the index is equal or greater 
           than the number of states is considered to be terminal
        2. It has no actions (not even invalid ones)
    */
    bool end_condition(State s) const 
        {return (size_t(s) >= mdp->size()) || (action_count(s) == 0);};

    /// State dependent action list 
    size_t action_count(State state) const 
        {return (*mdp)[state].size();};

    vector<Action> get_valid_actions(State state) const{
        vector<Action> valid_actions;
        const auto& mdpstate = (*mdp)[state];
        for(Action a=0l;a<(long)mdpstate.size();a++){
            if(mdpstate.is_valid(a)){
                valid_actions.push_back(a);
            }
        }
        return valid_actions;
    }

    /// Returns an action with the given index
    Action action(State, long index) const
        {return index;}

protected:
    /// Random number engine
    default_random_engine gen;

    /** MDP used for the simulation */
    shared_ptr<const MDP> mdp;
    
    /** Initial distribution */
    Transition initial;
};



/// Random (uniformly) policy to be used with the model simulator
using ModelRandomPolicy = RandomPolicy<ModelSimulator>;

/**
Randomized policy to be used with MDP model simulator.

In order to have a determinstic outcome of a simulation, one
needs to set also the seed of simulate and ModelSimulator.
*/ 
using ModelRandomizedPolicy = RandomizedPolicy<ModelSimulator>;

/// Deterministic policy to be used with MDP model simulator
using ModelDeterministicPolicy = DeterministicPolicy<ModelSimulator>;

/// Stochastic policy to be used with MDP model simulator
using ModelStochasticPolicy = StochasticPolicy<ModelSimulator>;

} // end namespace msen
} // end namespace craam

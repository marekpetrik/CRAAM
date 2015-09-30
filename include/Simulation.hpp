#pragma once

#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>
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
template <class Sim>
class Samples {


public:
    vector<DSample<Sim>> decsamples;
    vector<typename Sim::DState> initial;
    vector<ESample<Sim>> expsamples;

public:

    void add_dec(const DSample<Sim>& decsample){
        /**
         * Adds a sample starting in a decision state
         */
        this->decsamples.push_back(decsample);
    };

    void add_initial(const typename Sim::DState& decstate){
        /**
           Adds an initial state
         */
         this->initial.push_back(decstate);
    };

    void add_exp(const ESample<Sim>& expsample){
        /**
           Adds a sample starting in an expectation state
         */
        this->expsamples.push_back(expsample);
    };

    prec_t mean_return(prec_t discount){
        /**
           Computes the discounted mean return over all the
           samples
         *
           \param discount Discount factor
         */

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

    typename Sim::Action operator() (typename Sim::DState dstate){
        /**
           Returns the random action
         */
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

    typename Sim::Action operator() (typename Sim::DState dstate){
        /**
           Returns the random action
         */
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
template<class Sim,class SampleType=Samples<Sim>> unique_ptr<SampleType>
simulate_stateless( Sim& sim,
                    const function<typename Sim::Action(typename Sim::DState&)>& policy,
                    long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
                    random_device::result_type seed = random_device{}()){
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
        \param policy Policy function
        \param horizon Number of steps
        \param prob_term The probability of termination in each step

        \return Samples
     */

    unique_ptr<SampleType> samples(new SampleType());

    long transitions = 0;

    // initialize random numbers when appropriate
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    for(auto run=0l; run < runs; run++){

        typename Sim::DState&& decstate = sim.init_state();
        samples->add_initial(decstate);

        for(auto step=0l; step < horizon; step++){
            if(sim.end_condition(decstate))
                break;
            if(tran_limit > 0 && transitions > tran_limit)
                break;

            typename Sim::Action&& action = policy(decstate);
            typename Sim::EState&& expstate = sim.transition_dec(decstate,action);

            samples->add_dec(DSample<Sim> (decstate, action, expstate, step, run));

            auto&& rewardstate = sim.transition_exp(expstate);

            auto reward = rewardstate.first;
            decstate = rewardstate.second;

            samples->add_exp(ESample<Sim>(expstate, decstate, reward, 1.0, step, run));

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
    return samples;
}

} // end namespace msen
} // end namespace craam

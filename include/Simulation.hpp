#include <utility>
#include <vector>
#include <memory>
#include <random>
#include <functional>
#include <cmath>
#include <set>

#include "definitions.hpp"

using namespace std;

/*
 * Signature of static methods required for the simulator
 *
 * States and actions are passed by value
 *
 * DState init_state() const
 * EState transition_dec(DState, Action) const
 * pair<double,DState> transition_exp(EState) const
 * bool end_condition(DState) const
 * vector<Action> actions(DState)  const // needed for a random policy and value function policy
 * vector<Action> actions const          // an alternative when the actions are not state dependent
 */
template <class DecState,class ExpState>
struct ExpSample {
    /**
     * \brief Represents the transition from an expectation state to a
     * a decision state.
     */
    const ExpState expstate_from;
    const DecState decstate_to;
    const prec_t reward;
    const prec_t weight;
    const long step;
    const long run;

    ExpSample(const ExpState& expstate_from, const DecState& decstate_to,
              prec_t reward, prec_t weight, long step, long run):
        expstate_from(expstate_from), decstate_to(decstate_to),
        reward(reward), weight(weight), step(step), run(run)
                  {};
};

template <class DecState,class Action,class ExpState=pair<DecState,Action>>
struct DecSample {
    /**
     * \brief Represents the transition from a decision state to an
     * expectation state.
     */
    const DecState decstate_from;
    const Action action;
    const ExpState expstate_to;
    const long step;
    const long run;

    DecSample(const DecState& decstate_from, const Action& action,
              const ExpState& expstate_to, long step, long run):
        decstate_from(decstate_from), action(action),
        expstate_to(expstate_to), step(step), run(run)
        {};
};

template <class DecState,class Action,class ExpState=pair<DecState,Action>>
class Samples {
    /**
     * \brief General representation of samples
     */

public:
    vector<DecSample<DecState,Action,ExpState>> decsamples;
    vector<DecState> initial;
    vector<ExpSample<DecState,ExpState>> expsamples;

public:

    void add_dec(const DecSample<DecState,Action,ExpState>& decsample){
        /**
         * \brief Adds a sample starting in a decision state
         */
        this->decsamples.push_back(decsample);
    };

    void add_initial(const DecState& decstate){
        /**
         * \brief Adds an initial state
         */
         this->initial.push_back(decstate);
    };

    void add_exp(const ExpSample<DecState,ExpState>& expsample){
        /**
         * \brief Adds a sample starting in an expectation state
         */
        this->expsamples.push_back(expsample);
    };

    prec_t mean_return(prec_t discount){
        /**
         * \brief Computes the discounted mean return over all the
         * samples
         *
         * \param discount Discount factor
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

template<class Sim, class DState, class Action>
class RandomPolicySD {
    /**
     * \brief An object that behaves as a random policy for problems
     * with state-dependent actions.
     */

private:

    const Sim& sim;
    default_random_engine gen;

public:

    RandomPolicySD(const Sim& sim, random_device::result_type seed = random_device{}()) : sim(sim), gen(seed)
    {};

    Action operator() (DState dstate){
        /**
         * \brief Returns the random action
         */
        const vector<Action>&& actions = sim.actions(dstate);

        auto actioncount = actions.size();
        uniform_int_distribution<> dst(0,actioncount-1);

        return actions[dst(gen)];
    };
};

template<class Sim, class DState, class Action>
class RandomPolicySI {
    /**
     * \brief An object that behaves as a random policy for problems
     * with state-dependent actions.
     * 
     * The actions are copied internally.
     */

private:

    const Sim& sim;
    default_random_engine gen;
    const vector<Action> actions;   // list of actions is constant

public:

    RandomPolicySI(const Sim& sim, random_device::result_type seed = random_device{}()) 
        : sim(sim), gen(seed), actions(sim.actions())
    {};

    Action operator() (DState dstate){
        /**
         * \brief Returns the random action
         */
        auto actioncount = actions.size();
        uniform_int_distribution<> dst(0,actioncount-1);

        return actions[dst(gen)];
    };
};

//-----------------------------------------------------------------------------------
template<class DState,class Action,class EState = pair<DState,Action>>

unique_ptr<Samples<DState,Action,EState>>
simulate_stateless(auto& sim, const function<Action(DState&)>& policy,
                   long horizon, long runs, long tran_limit=-1, prec_t prob_term=0.0,
                   random_device::result_type seed = random_device{}()){
    /** \brief Runs the simulator and generates samples. A simulator with no state
     *
     * \param sim Simulator that holds the properties needed by the simulator
     * \param policy Policy function
     * \param horizon Number of steps
     * \param prob_term The probability of termination in each step
     * \return Samples
     */

    unique_ptr<Samples<DState,Action,EState>> samples(new Samples<DState,Action,EState>());

    long transitions = 0;

    // initialize random numbers when appropriate
    default_random_engine generator(seed);
    uniform_real_distribution<double> distribution(0.0,1.0);

    for(auto run=0l; run < runs; run++){

        DState&& decstate = sim.init_state();
        samples->add_initial(decstate);

        for(auto step=0l; step < horizon; step++){
            if(sim.end_condition(decstate))
                break;
            if(tran_limit > 0 && transitions > tran_limit)
                break;

            Action&& action = policy(decstate);
            EState&& expstate = sim.transition_dec(decstate,action);

            samples->add_dec(DecSample<DState,Action,EState>
                                (decstate, action, expstate, step, run));

            auto&& rewardstate = sim.transition_exp(expstate);

            auto reward = rewardstate.first;
            decstate = rewardstate.second;

            samples->add_exp(ExpSample<DState,EState>(expstate, decstate, reward, 1.0, step, run));

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
};

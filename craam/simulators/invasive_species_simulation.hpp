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

#include "../Samples.hpp"
#include "../definitions.hpp"

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

/**
A policy for invasive species management that depends on the population threshold & ctrol probability.

\tparam Sim Simulator class for which the policy is to be constructed.
            Must implement an instance method actions(State).
 */
template<class Sim>
class InvasiveSpeciesPolicy{

public:
    using State = typename Sim::State;
    using Action = typename Sim::Action;

    InvasiveSpeciesPolicy(const Sim& sim, long threshold_control=0, prec_t prob_control = 0.5,
                    random_device::result_type seed = random_device{}()) :
                sim(sim), threshold_control(threshold_control), prob_control(prob_control), gen(seed){
                control_distribution = binomial_distribution<int>(1,prob_control);
    }

    /** provides a control action depending on the current population level. If the population level is below a certain threshold,
     * the policy is not to take the control measure. Otherwise, it takes a control measure with a specific probability (which introduces
     * randomness in the policy).
    */
    long operator() (long current_state){
		if (current_state>=threshold_control)
			return control_distribution(gen);
		return 0;
    }

private:
    /// Internal reference to the originating simulator
    const Sim& sim;
    long threshold_control;
    prec_t prob_control;
    /// Random number engine
    default_random_engine gen;
    binomial_distribution<int> control_distribution;
};


/**
A simulator to generate invasive species simulation data.
*/
class InvasiveSpeciesSimulator{

public:
    /// Type of states
    typedef long State;
    /// Type of actions
    typedef long Action;

    /**
    Build a model simulator for invasive species.

	\param initial_population Startning population to start the simulation
	\param carrying_capacity Maximum possible amount of population
	\param mean_growth_rate Mean of the population growth rate
	\param std_growth_rate Standard deviation of the growth rate
	\param std_observation Standard deviation for the observation from the actual underlying population
	\param beta_1 Coefficient of effectiveness
	\param beta_2 Coefficient of effectiveness for the quadratic term
	\param n_hat Threhold when the quadratic effects kick in
	\param seed Seed for random number generation
    */
    InvasiveSpeciesSimulator(long initial_population, long carrying_capacity, prec_t mean_growth_rate, prec_t std_growth_rate,
                       prec_t std_observation, prec_t beta_1, prec_t beta_2, long n_hat, random_device::result_type seed = random_device{}()) :
        initial_population(initial_population), carrying_capacity(carrying_capacity), mean_growth_rate(mean_growth_rate),
        std_growth_rate(std_growth_rate), std_observation(std_observation), beta_1(beta_1), beta_2(beta_2), n_hat(n_hat), gen(seed) {}

    long init_state() const{
        return initial_population;
    }

    bool end_condition(State s) const{
		return initial_population<0;
	}

    /**
    Returns a sample of the reward and a population level following an action & current population level
    
    When the treatment action is not chosen (action=0), 
    then growth_rate = max(0,Normal(mean_growth_rate, std_growth_rate)), In this case, the growth rate is independent of the current population level.

	When the treatment is applied (action=1), 
	then the growth rate depends on the current population level & the beta_x parameters come into play.
	
	The next population is obtained using the logistic population growth model: 
		min(carrying_capacity, growth_rate * current_population * (carrying_capacity-current_population)/carrying_capacity)
    
    The observed population deviates from the actual population & is stored in observed_population.
    
    The treatment action carries a fixed cost control_reward (or a negative reward of -4000) of applying the treatment. There is a variable population dependent 
    cost invasive_reward (-1) that represents the economic (or ecological) damage of the invasive species.
    
	\param current_population Current population level
	\param action Whether the control measure should be taken or not

	\returns a pair of reward & next population level
    */
    pair<prec_t,long> transition(long current_population, long action){
        assert(current_population >= 0 );
		
        normal_distribution<prec_t> growth_rate_distribution(max(0.0, mean_growth_rate - action*current_population*beta_1 - action*pow(max(current_population-n_hat,0l), 2)*beta_2 ), std_growth_rate);
		
        prec_t growth_rate = growth_rate_distribution(gen);
        long next_population = max(0l, min(carrying_capacity, (long)growth_rate * current_population * (carrying_capacity-current_population)/carrying_capacity));
        normal_distribution<prec_t> observation_distribution(next_population, std_observation);
        long observed_population = max(0l, (long) observation_distribution(gen));
        prec_t reward = next_population * (-1) + action * (-4000);
        return make_pair(reward, observed_population);
    }

protected:
    /// Random number engine
    long initial_population, carrying_capacity;
    prec_t mean_growth_rate, std_growth_rate, std_observation, beta_1, beta_2;
    long n_hat;
    default_random_engine gen;
};

///Invasive Species policy to be used
using ModelInvasiveSpeciesPolicy = InvasiveSpeciesPolicy<InvasiveSpeciesSimulator>;

} // end namespace msen
} // end namespace craam

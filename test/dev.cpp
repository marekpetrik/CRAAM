// Simple development tests

#include "craam/ImMDP.hpp"
#include "craam/Simulation.hpp"
#include "craam/Samples.hpp"
#include "craam/algorithms/values.hpp"

#include "rm/range.hpp"

#include <iostream>
#include <iterator>
#include <random>
#include <cmath>
#include <cassert>

#include <boost/functional/hash.hpp>
#include <iostream>
#include <iterator>


using namespace std;
using namespace craam;
using namespace craam::impl;
using namespace util::lang;

template<class T>
void print_vector(vector<T> vec){
    for(auto&& p : vec){
        cout << p << " ";
    }
}

/**
A simple simulator class. The state represents a position in a chain
and actions move it up and down. The reward is equal to the position.

Representation
~~~~~~~~~~~~~~
- State: position (int)
- Action: change (int)
*/
class Counter{
private:
    default_random_engine gen;
    bernoulli_distribution d;
    const vector<int> actions_list;
    const int initstate;

public:
    using State = int;
    using Action = int;

    /**
    Define the success of each action
    \param success The probability that the action is actually applied
    */
    Counter(double success, int initstate, random_device::result_type seed = random_device{}())
        : gen(seed), d(success), actions_list({1,-1}), initstate(initstate) {};

    int init_state() const {
        return initstate;
    }

    pair<double,int> transition(int pos, int action) {
        int nextpos = d(gen) ? pos + action : pos;
        return make_pair((double) pos, nextpos);
    }

    bool end_condition(const int state){
        return false;
    }

    int action(State state, long index) const{
        return actions_list[index];
    }

    const vector<int>& get_valid_actions(int state) const{
        return actions_list; 
    }

    size_t action_count(State) const{return actions_list.size();};
};


/** A counter that terminates at either end as defined by the end state */
class CounterTerminal : public Counter {
public:
    int endstate;

    CounterTerminal(double success, int initstate, int endstate, random_device::result_type seed = random_device{}())
        : Counter(success, initstate, seed), endstate(endstate) {};

    bool end_condition(const int state){
        return (abs(state) >= endstate);
    }
};
// Hash function for the Counter / CounterTerminal EState above
namespace std{
    template<> struct hash<pair<int,int>>{
        size_t operator()(pair<int,int> const& s) const{
            boost::hash<pair<int,int>> h;
            return h(s);
        };
    };
}

using namespace craam::msen;

int main(void){

    const int terminal_state = 8;
    const prec_t discount = 0.9;

    CounterTerminal sim(0.9,0,terminal_state,1);
    RandomPolicy<CounterTerminal> random_pol(sim);

    auto samples = make_samples<CounterTerminal>();
    simulate(sim,samples,random_pol,100,100);
    simulate(sim,samples,[](int){return 1;},10,20);
    simulate(sim,samples,[](int){return -1;},10,20);

    SampleDiscretizerSI<typename CounterTerminal::State, typename CounterTerminal::Action> sd;
    // initialize action values
    sd.add_action(-1); sd.add_action(+1);
    //initialize state values
    for(auto i : util::lang::range(-terminal_state,terminal_state)) sd.add_state(i);

    sd.add_samples(samples);

    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());
    auto mdp = smdp.get_mdp();
    auto&& initial = smdp.get_initial();

    auto&& sol = algorithms::solve_mpi(*mdp,discount);

    cout << "Optimal policy: "; print_vector(sol.policy); cout << "Return " <<  sol.total_return(initial) << endl;

    // define observations
    indvec observations(mdp->state_count(), -1);
    size_t last_obs(0), inobs(0);
    cout << "Observations: " << mdp->state_count() << " states  ";
    for(auto i : util::lang::range(size_t(0), mdp->state_count())){
        // check if this is a terminal state
        if(mdp->get_state(i).action_count() == 0 || inobs >= 2){
            if(inobs > 0 && mdp->get_state(i).action_count() == 0){
                last_obs++;
            }
            observations[i] = last_obs++;
            inobs = 0;
        }else {
            observations[i] = last_obs;
            inobs++;
        }
        cout << observations[i] << " ";
    }
    cout << endl;

    MDPI_R mdpi(mdp, observations, initial);
    auto&& randompolicy = mdpi.random_policy(25);

    auto isol = mdpi.solve_reweighted(0, discount, randompolicy);

    isol = mdpi.solve_reweighted(10, discount, randompolicy);

    auto sol_impl = solve_mpi(*mdp, discount, numvec(0), mdpi.obspol2statepol(isol));

    cout << "Implementable pol: "; print_vector(isol);
    cout << "  Return: " << sol_impl.total_return(initial) << endl;

    cout << "Generating implementable policies (randomly) ..." << endl;

    auto max_return = 0.0;
    indvec max_pol(mdpi.obs_count(),-1);

    for(auto i : util::lang::range(0,200)){
        (void)(i);
        auto rand_pol = mdpi.random_policy();

        auto ret = solve_mpi(*mdp, discount, numvec(0), mdpi.obspol2statepol(rand_pol)).total_return(initial);

        if(ret > max_return){
            max_pol = rand_pol;
            max_return = ret;
        }

    }

    cout << "Maximal return " << max_return << endl;
    cout << "Best policy: ";
    print_vector(max_pol);
    cout << endl;

    return 0;

}

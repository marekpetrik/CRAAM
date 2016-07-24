// Simple development tests

#include <iostream>
#include <iterator>
#include <random>
#include <cmath>
#include <cassert>

#include <boost/functional/hash.hpp>
#include <iostream>
#include <iterator>

#include "ImMDP.hpp"
#include "Simulation.hpp"
#include "Samples.hpp"
#include "cpp11-range-master/range.hpp"

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
//
//int main_im(){
//    const double discount = 0.9;
//
//    cout << "Running ... " << endl;
//
//    auto mdpi = MDPI_R::from_csv_file("mdp.csv","observ.csv","initial.csv");
//
//    cout << "States: " << mdpi->state_count() << " Observations: " << mdpi->obs_count() << endl;
//    cout << "Solving MDP ..." << endl;
//
//    auto mdp = mdpi->get_mdp();
//    auto&& sol = mdp->mpi_jac_ave(numvec(0), discount);
//    auto&& initial = mdpi->get_initial();
//
//    cout << "Return: " << sol.total_return(initial) << endl;
//    cout << "Policy: ";
//    print_vector(sol.policy);
//    cout << endl;
//
//    // check that the policy is correct
//    auto res = mdp->assert_policy_correct(indvec(mdp->state_count(), 0), indvec(mdp->state_count(), 0));
//    assert(res == -1);
//
//    auto sol_base = mdp->vi_jac_fix(numvec(0),discount,indvec(mdp->state_count(), 0),
//                                    indvec(mdp->state_count(), 0));
//    cout << "Baseline policy return: " << sol_base.total_return(initial) << endl;
//
//    cout << "Solving constrained MDP ... " << endl;
//
//    for(auto i : range(0,5)){
//        auto pol = mdpi->solve_reweighted(i,0.9);
//        cout << "Iteration: " << i << "  :  ";
//        print_vector(pol);
//        cout << endl;
//    }
//
//    auto pol = mdpi->solve_reweighted(10,discount);
//
//    auto sol_impl = mdp->vi_jac_fix(numvec(0),discount, mdpi->obspol2statepol(pol),
//                    indvec(mdp->state_count(), 0));
//
//    cout << "Return implementable: " << sol_impl.total_return(initial) << endl;
//
//    cout << "Generating implementable policies (randomly) ..." << endl;
//
//    auto max_return = 0.0;
//    indvec max_pol(mdpi->obs_count(),-1);
//
//    for(auto i : range(0,20000)){
//        auto rand_pol = mdpi->random_policy();
//
//        auto ret = mdp->vi_jac_fix(numvec(0),discount, mdpi->obspol2statepol(rand_pol),
//                    indvec(mdp->state_count(), 0)).total_return(initial);
//
//        if(ret > max_return){
//            max_pol = rand_pol;
//            max_return = ret;
//        }
//
//        //cout << "index " << i << " return " << ret << endl;
//    }
//
//    cout << "Maximal return " << max_return << endl;
//    cout << "Best policy: ";
//    print_vector(max_pol);
//    cout << endl;
//
//    return 0;
//
//}

/**
A simple simulator class. The state represents a position in a chain
and actions move it up and down. The reward is equal to the position.

Representation
~~~~~~~~~~~~~~
- Decision state: position (int)
- Action: change (int)
- Expectation state: position, action (int,int)
*/
class Counter{
private:
    default_random_engine gen;
    bernoulli_distribution d;
    const vector<int> actions_list;
    const int initstate;

public:
    typedef int DState;
    typedef pair<int,int> EState;
    typedef int Action;

    /**
    Define the success of each action
    \param success The probability that the action is actually applied
    */
    Counter(double success, int initstate, random_device::result_type seed = random_device{}())
        : gen(seed), d(success), actions_list({1,-1}), initstate(initstate) {};

    int init_state() const {
        return initstate;
    }

    pair<int,int> transition_dec(int state, int action) const{
        return make_pair(state,action);
    }

    pair<double,int> transition_exp(const pair<int,int> expstate) {
        int pos = expstate.first;
        int act = expstate.second;

        int nextpos = d(gen) ? pos + act : pos;
        return make_pair((double) pos, nextpos);
    }

    bool end_condition(const int state){
        return false;
    }

    vector<int> actions(int) const{
        return actions_list;
    }

    vector<int> actions() const{
        return actions_list;
    }
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
    RandomPolicySI<CounterTerminal> random_pol(sim);

    Samples<CounterTerminal> samples;
    simulate_stateless(sim,samples,random_pol,100,100);
    simulate_stateless(sim,samples,[](int){return 1;},10,20);
    simulate_stateless(sim,samples,[](int){return -1;},10,20);

    SampleDiscretizerSI<CounterTerminal> sd;
    // initialize action values
    sd.add_action(-1); sd.add_action(+1);
    //initialize state values
    for(auto i : range(-terminal_state,terminal_state)) sd.add_dstate(i);

    sd.add_samples(samples);

    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());
    auto mdp = smdp.get_mdp();
    auto&& initial = smdp.get_initial();

    auto&& sol = mdp->mpi_jac_ave(numvec(0),discount);

    cout << "Optimal policy: "; print_vector(sol.policy); cout << "Return " <<  sol.total_return(initial) << endl;

    // define observations
    indvec observations(mdp->state_count(), -1);
    size_t last_obs(0), inobs(0);
    cout << "Observations: " << mdp->state_count() << " states  ";
    for(auto i : range(size_t(0), mdp->state_count())){
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

    auto sol_impl = mdp->vi_jac_fix(numvec(0),discount, mdpi.obspol2statepol(isol),
                    indvec(mdp->state_count(), 0));

    cout << "Implementable pol: "; print_vector(isol);
    cout << "  Return: " << sol_impl.total_return(initial) << endl;

    cout << "Generating implementable policies (randomly) ..." << endl;

    auto max_return = 0.0;
    indvec max_pol(mdpi.obs_count(),-1);

    for(auto i : range(0,200)){
        auto rand_pol = mdpi.random_policy();

        auto ret = mdp->vi_jac_fix(numvec(0),discount, mdpi.obspol2statepol(rand_pol),
                    indvec(mdp->state_count(), 0)).total_return(initial);

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

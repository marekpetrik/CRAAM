#include <iostream>
#include <sstream>
#include <random>
#include <utility>
#include <functional>

#include <boost/functional/hash.hpp>
#include "cpp11-range-master/range.hpp"

#include "Simulation.hpp"

using namespace std;
using namespace craam;
using namespace craam::msen;
using namespace util::lang;


#define BOOST_TEST_DYN_LINK
//#define BOOST_TEST_MAIN

#define CHECK_CLOSE_COLLECTION(aa, bb, tolerance) { \
    using std::distance; \
    using std::begin; \
    using std::end; \
    auto a = begin(aa), ae = end(aa); \
    auto b = begin(bb); \
    BOOST_REQUIRE_EQUAL(distance(a, ae), distance(b, end(bb))); \
    for(; a != ae; ++a, ++b) { \
        BOOST_CHECK_CLOSE(*a, *b, tolerance); \
    } \
}

//#define BOOST_TEST_MODULE MainModule
#include <boost/test/unit_test.hpp>

struct TestState{
    int index;

    TestState(int i){
        this->index = i;
    };
};


class TestSim {

public:

    typedef TestState State;
    typedef int Action;

    TestState init_state() const{
        return TestState(1);
    }

    pair<double,TestState> transition(TestState, int) const{
        return pair<double,TestState>(1.0,TestState(1));
    };

    bool end_condition(TestState) const{
        return false;
    };

    vector<int> actions(TestState) const{
        return vector<int>{1};
    };

};

int test_policy(TestState){
    return 0;
}

BOOST_AUTO_TEST_CASE(basic_simulation) {
    TestSim sim;

    auto samples = simulate_stateless<TestSim>(sim, test_policy, 10,5);
    BOOST_CHECK_EQUAL(samples.size(), 50);
}


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



BOOST_AUTO_TEST_CASE(simulation_multiple_counter_sd ) {
    Counter sim(0.9,0,1);

    RandomPolicySD<Counter> random_pol(sim,1);
    auto samples = simulate_stateless(sim,random_pol,20,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), -3.51759102217019, 0.0001);

    samples = simulate_stateless(sim,random_pol,1,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 0, 0.0001);

    Counter sim2(0.9,3,1);
    samples = simulate_stateless(sim2,random_pol,1,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 3, 0.0001);
}

BOOST_AUTO_TEST_CASE(simulation_multiple_counter_si ) {
    Counter sim(0.9,0,1);

    RandomPolicySI<Counter> random_pol(sim,1);
    auto samples = simulate_stateless(sim,random_pol,20,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), -3.51759102217019, 0.0001);

    samples = simulate_stateless(sim,random_pol,1,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 0, 0.0001);

    Counter sim2(0.9,3,1);
    samples = simulate_stateless(sim2,random_pol,1,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 3, 0.0001);
}

BOOST_AUTO_TEST_CASE(construct_mdp_from_samples_sd_pol){

    CounterTerminal sim(0.9,0,10,1);

    RandomPolicySI<CounterTerminal> random_pol(sim,1);
    auto samples = simulate_stateless(sim,random_pol,20,20);

    SampleDiscretizerSD<CounterTerminal> sd;
    sd.add_samples(samples);

    BOOST_CHECK_EQUAL(samples.get_initial().size(), sd.get_discrete()->get_initial().size());
    BOOST_CHECK_EQUAL(samples.size(), sd.get_discrete()->size());

    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());

    shared_ptr<const MDP> mdp = smdp.get_mdp();

    // check that the number of actions is correct (2)
    for(auto i : range((size_t)0, mdp->state_count())){
        if(mdp->get_state(i).action_count() > 0)
            BOOST_CHECK_LE(mdp->get_state(i).action_count(), 2);
    }

    auto&& sol = mdp->mpi_jac(Uncertainty::Average,0.9);

    BOOST_CHECK_CLOSE(sol.total_return(smdp.get_initial()), 47.6799, 1e-3);
}


BOOST_AUTO_TEST_CASE(construct_mdp_from_samples_si_pol){

    CounterTerminal sim(0.9,0,10,1);
    RandomPolicySI<CounterTerminal> random_pol(sim,1);

    Samples<CounterTerminal> samples;
    simulate_stateless(sim,samples,random_pol,50,50);
    simulate_stateless(sim,samples,[](int){return 1;},10,20);
    simulate_stateless(sim,samples,[](int){return -1;},10,20);

    SampleDiscretizerSI<CounterTerminal> sd;
    sd.add_samples(samples);

    BOOST_CHECK_EQUAL(samples.get_initial().size(), sd.get_discrete()->get_initial().size());
    BOOST_CHECK_EQUAL(samples.size(), sd.get_discrete()->size());


    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());

    shared_ptr<const MDP> mdp = smdp.get_mdp();

    // check that the number of actions is correct (2)
    for(auto i : range((size_t)0, mdp->state_count())){
        if(mdp->get_state(i).action_count() > 0)
            BOOST_CHECK_EQUAL(mdp->get_state(i).action_count(), 2);
    }

    auto&& sol = mdp->mpi_jac(Uncertainty::Average,0.9);

    BOOST_CHECK_CLOSE(sol.total_return(smdp.get_initial()), 51.313973, 1e-3);
}

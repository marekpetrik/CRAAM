#include <iostream>
#include <sstream>
#include <random>
#include <utility>
#include <functional>
#include "Simulation.hpp"

using namespace std;

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

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

#define BOOST_TEST_MODULE MainModule
#include <boost/test/unit_test.hpp>


struct TestDState{
    int index;

    TestDState(int i){
        this->index = i;
    };
};


struct TestEState{
    int index;

    TestEState(int i){
        this->index = i;
    };
};


class TestSim {

public:

    TestDState init_state() const{
        return TestDState(1);
    }

    TestEState transition_dec(TestDState, int) const{
        return TestEState(1);
    };

    pair<double,TestDState> transition_exp(TestEState) const{
        return pair<double,TestDState>(1.0,TestDState(1));
    };

    bool end_condition(TestDState) const{
        return false;
    };

    vector<int> actions(TestDState) const{
        return vector<int>{1};
    };

};

int test_policy(TestDState){
    return 0;
};

BOOST_AUTO_TEST_CASE( basic_simulation ) {
    TestSim sim;

    auto samples = simulate_stateless<TestDState,int,TestEState>(sim, test_policy, 10,5);
    BOOST_CHECK_EQUAL(samples->decsamples.size(),50);
    BOOST_CHECK_EQUAL(samples->expsamples.size(),50);
}

class Counter{
    /**
     * Decision state: position
     * Expectation state: position, action
     */
public:

    default_random_engine gen;
    bernoulli_distribution d;
    const vector<int> actions_list;;

    Counter(double success, random_device::result_type seed = random_device{}())
        : gen(seed), d(success), actions_list({1,-1}) {
        /** \brief Define the success of each action
         * \param success The probability that the action is actually applied
         */
    };

    int init_state() const {
        return 0;
    };

    pair<int,int> transition_dec(int state, int action) const{
        return make_pair(state,action);
    };

    pair<double,int> transition_exp(const pair<int,int> expstate) {
        int pos = expstate.first;
        int act = expstate.second;

        //cout << "(" << pos << "," << act << ") ";

        int nextpos = d(gen) ? pos + act : pos;

        return make_pair((double) pos, nextpos);
    };

    bool end_condition(const int state){
        return false;
    }

    vector<int> actions(int) const{
        return actions_list;
    };
};



BOOST_AUTO_TEST_CASE( simulation_multiple_counter ) {
    Counter sim(0.9,1);

    RandomPolicy<Counter,int,int> random_pol(sim,1);
    auto samples = simulate_stateless<int,int,pair<int,int>>(sim,random_pol,20,20);

    BOOST_CHECK_CLOSE(samples->mean_return(0.9), -3.51759102217019, 0.0001);
}

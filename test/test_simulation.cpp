#include <iostream>
#include <sstream>
#include <random>
#include <utility>

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
//#include <boost/test/included/unit_test.hpp>
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

    TestEState transition_dec(const TestDState&, const int&) const{
        return TestEState(1);
    };

    pair<double,TestDState> transition_exp(const TestEState&) const{
        return pair<double,TestDState>(1.0,TestDState(1));
    };

    bool end_condition(const TestDState&) const{
        return false;
    };

    vector<int> actions(const TestDState&) const{
        return vector<int>{1};
    };

};

int test_policy(TestDState){
    return 0;
};

BOOST_AUTO_TEST_CASE( basic_simulation ) {
    TestSim sim;

    auto samples = simulate_stateless<TestDState,int,TestEState,TestSim,test_policy>(sim, 10,5);

    cout << samples->decsamples.size() << endl;
}


class Counter{
    /**
     * Decision state: position
     * Expectation state: position, action
     */
public:

    default_random_engine gen;
    bernoulli_distribution d;
    const array<int,3> actions_list;;


    Counter(double success) : gen(), d(success), actions_list({0,1,-1}) {
        /** \brief Define the success of each action
         * \param success The probability that the action is actually applied
         */
    };

    int init_state() const {
        return 0;
    };

    pair<int,int> transition_dec(const int state, const int action) const{
        return make_pair(state,action);
    };

    pair<double,int> transition_exp(const pair<int,int> expstate) {
        int pos = expstate.first;
        int act = expstate.second;

        int nextpos = d(gen) ? pos + act : pos;

        return make_pair((double) pos, nextpos);
    };

    bool end_condition(const int state){
        return false;
    }

    const array<int,3> actions(const TestDState&) const{
        return actions_list;
    };

};



BOOST_AUTO_TEST_CASE( counter_simulation ) {
    Counter sim(0.2);

    auto samples = simulate_stateless<int,int,pair<int,int>,Counter,test_policy>(sim, 10,5);

    cout << samples->decsamples.size() << endl;
}


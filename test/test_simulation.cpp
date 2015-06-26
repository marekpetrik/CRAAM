#include <iostream>
#include <sstream>

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
private:
    TestSim();

public:

    static TestDState init_state(){
        return TestDState(1);
    }

    static TestEState transition_dec(const TestDState&, const int&){
        return TestEState(1);
    };

    static pair<double,TestDState> transition_exp(const TestEState&){
        return pair<double,TestDState>(1.0,TestDState(1));
    };

    static bool end_condition(const TestDState&){
        return false;
    };

    static vector<int> actions(const TestDState&){
        return vector<int>{1};
    };

};

int test_policy(TestDState){
    return 0;
}


BOOST_AUTO_TEST_CASE( basic_simulation ) {

    auto samples = simulate_stateless<TestDState,int,TestEState,TestSim,test_policy>(10,5);

    cout << samples->decsamples.size() << endl;


}

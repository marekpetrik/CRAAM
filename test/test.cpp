
#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <utility>


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


#include "core_tests.hpp"
#include "simulation_tests.hpp"
#include "implementable_tests.hpp"

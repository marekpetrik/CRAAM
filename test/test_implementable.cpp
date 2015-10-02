#include "RMDP.hpp"
#include "ImMDP.hpp"
#include "Transition.hpp"

#include <iostream>

using namespace std;
using namespace craam;
using namespace craam::impl;


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

#define BOOST_TEST_DYN_LINK
//#define BOOST_TEST_MAIN

//#define BOOST_TEST_MODULE MainModule
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( simple_construct_mdpi ) {

    shared_ptr<RMDP> mdp(new RMDP());
    vector<long> observations({0,0});
    Transition initial(vector<long>{0,1},vector<prec_t>{0.5,0.5},vector<prec_t>{0,0});

    mdp->add_transition_d(0,0,1,1.0,1.0);
    mdp->add_transition_d(1,0,0,1.0,1.0);
    BOOST_CHECK_EQUAL(mdp->state_count(), 2);

    MDPI im(const_pointer_cast<const RMDP>(mdp), observations, initial);

    MDPI im2(*mdp,observations,initial);

    // check that we really have a copy
    mdp->add_transition_d(1,0,2,1.0,1.0);
    BOOST_CHECK_EQUAL(mdp->state_count(), 3);
    BOOST_CHECK_EQUAL(im.mdp->state_count(), 3);
    BOOST_CHECK_EQUAL(im2.mdp->state_count(), 2);
}


BOOST_AUTO_TEST_CASE( simple_construct_mdpi_r ) {

    shared_ptr<RMDP> mdp(new RMDP());
    vector<long> observations({0,0});
    Transition initial(vector<long>{0,1},vector<prec_t>{0.5,0.5},vector<prec_t>{0,0});

    mdp->add_transition_d(0,0,1,1.0,1.0);
    mdp->add_transition_d(1,0,0,1.0,2.0);

    MDPI_R imr(const_pointer_cast<const RMDP>(mdp), observations, initial);

    const RMDP& rmdp = imr.get_robust_mdp();

    BOOST_CHECK_EQUAL(rmdp.state_count(), 1);
    BOOST_CHECK_EQUAL(rmdp.action_count(0), 1);
    BOOST_CHECK_EQUAL(rmdp.outcome_count(0,0), 2);

    vector<prec_t> iv(rmdp.state_count(),0.0);

    Solution&& so = rmdp.mpi_jac_opt(iv,0.9,100,0.0,10,0.0);
    BOOST_CHECK_CLOSE(so.valuefunction[0], 20, 1e-3);

    Solution&& sr = rmdp.mpi_jac_rob(iv,0.9,100,0.0,10,0.0);
    BOOST_CHECK_CLOSE(sr.valuefunction[0], 10, 1e-3);
}

BOOST_AUTO_TEST_CASE( small_construct_mdpi_r ) {

    shared_ptr<RMDP> mdp(new RMDP());
    vector<long> observations({0,0,1});
    Transition initial(vector<long>{0,1,2},vector<prec_t>{1.0/3.0,1.0/3.0,1.0/3.0},
                        vector<prec_t>{0,0,0});

    // action 0
    mdp->add_transition_d(0,0,0,0.5,1.0);
    mdp->add_transition_d(0,0,1,0.5,1.0);

    mdp->add_transition_d(1,0,0,0.5,2.0);
    mdp->add_transition_d(1,0,1,0.5,2.0);

    mdp->add_transition_d(2,0,2,1.0,1.2);

    // action 1
    mdp->add_transition_d(0,1,2,1.0,1.2);
    mdp->add_transition_d(1,1,2,1.0,1.2);

    MDPI_R imr(const_pointer_cast<const RMDP>(mdp), observations, initial);

    const RMDP& rmdp = imr.get_robust_mdp();

    BOOST_CHECK_EQUAL(rmdp.state_count(), 2);
    BOOST_CHECK_EQUAL(rmdp.action_count(0), 2);
    BOOST_CHECK_EQUAL(rmdp.action_count(1), 1);
    BOOST_CHECK_EQUAL(rmdp.outcome_count(0,0), 2);
    BOOST_CHECK_EQUAL(rmdp.outcome_count(0,1), 2);
    BOOST_CHECK_EQUAL(rmdp.outcome_count(1,0), 1);

    vector<prec_t> iv(rmdp.state_count(),0.0);


    vector<prec_t> target_v_opt{20.0,12.0};
    vector<prec_t> target_v_rob{12.0,12.0};

    Solution&& so = rmdp.mpi_jac_opt(iv,0.9,100,0.0,10,0.0);
    CHECK_CLOSE_COLLECTION(so.valuefunction, target_v_opt, 1e-3);

    Solution&& sr = rmdp.mpi_jac_rob(iv,0.9,100,0.0,10,0.0);
    CHECK_CLOSE_COLLECTION(sr.valuefunction, target_v_rob, 1e-3);
}

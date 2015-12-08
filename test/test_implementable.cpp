#include "RMDP.hpp"
#include "ImMDP.hpp"
#include "Transition.hpp"

#include <iostream>
#include <iterator>

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
    BOOST_CHECK_EQUAL(im.get_mdp()->state_count(), 3);
    BOOST_CHECK_EQUAL(im2.get_mdp()->state_count(), 2);
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
    BOOST_CHECK_EQUAL(rmdp.get_state(0).action_count(), 1);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).get_action(0).outcome_count(), 2);

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

    BOOST_TEST_CHECKPOINT("Constructing MDPI_R.");
    MDPI_R imr(const_pointer_cast<const RMDP>(mdp), observations, initial);

    const RMDP& rmdp = imr.get_robust_mdp();

    BOOST_TEST_CHECKPOINT("Checking MDP properties.");
    BOOST_CHECK_EQUAL(rmdp.state_count(), 2);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).action_count(), 2);
    BOOST_CHECK_EQUAL(rmdp.get_state(1).action_count(), 1);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).get_action(0).outcome_count(), 2);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).get_action(1).outcome_count(), 2);
    BOOST_CHECK_EQUAL(rmdp.get_state(1).get_action(0).outcome_count(), 1);

    vector<prec_t> iv(rmdp.state_count(),0.0);

    vector<prec_t> target_v_opt{20.0,12.0};
    vector<prec_t> target_v_rob{12.0,12.0};

    BOOST_TEST_CHECKPOINT("Solving RMDP");
    Solution&& so = rmdp.mpi_jac_opt(iv,0.9,100,0.0,10,0.0);
    CHECK_CLOSE_COLLECTION(so.valuefunction, target_v_opt, 1e-3);

    Solution&& sr = rmdp.mpi_jac_rob(iv,0.9,100,0.0,10,0.0);
    CHECK_CLOSE_COLLECTION(sr.valuefunction, target_v_rob, 1e-3);
}

BOOST_AUTO_TEST_CASE( small_reweighted_solution ) {

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

    BOOST_TEST_CHECKPOINT("Constructing MDPI_R.");
    MDPI_R imr(const_pointer_cast<const RMDP>(mdp), observations, initial);

    BOOST_TEST_CHECKPOINT("Solving MDPI_R.");
    auto&& pol = imr.solve_reweighted(10, 0.9);

    indvec polvec{0,0};
    BOOST_CHECK_EQUAL_COLLECTIONS(pol.begin(), pol.end(),polvec.begin(),polvec.end());

    //ostream_iterator<prec_t> output(cout, ", ");
    //copy(pol.begin(), pol.end(), output);
}

BOOST_AUTO_TEST_CASE(simple_mdpo_save_load_save_load) {
    RMDP rmdp1(0);

    rmdp1.add_transition_d(0,1,1,1,0);
    rmdp1.add_transition_d(1,1,2,1,0);
    rmdp1.add_transition_d(2,1,2,1,1.1);

    rmdp1.add_transition_d(0,0,0,1,0);
    rmdp1.add_transition_d(1,0,0,1,1);
    rmdp1.add_transition_d(2,0,1,1,1);

    Transition initial(indvec{0,1,2},numvec{1.0/3.0,1.0/3.0,1/3.0});
    indvec state2obs{0,0,1};

    MDPI mdpi1(rmdp1, state2obs, initial);

    stringstream store1, store2, store3;

    mdpi1.to_csv(store1,store2,store3);
    store1.seekg(0); store2.seekg(0); store3.seekg(0);

    auto&& string11 = store1.str();
    auto&& string12 = store2.str();
    auto&& string13 = store3.str();

    auto mdpi2 = MDPI::from_csv(store1,store2,store3);

    stringstream store21, store22, store23;

    mdpi2->to_csv(store21, store22, store23);

    auto&& string21 = store21.str();
    auto&& string22 = store22.str();
    auto&& string23 = store23.str();

    BOOST_CHECK_EQUAL(string11, string21);
    BOOST_CHECK_EQUAL(string12, string22);
    BOOST_CHECK_EQUAL(string13, string23);
}

BOOST_AUTO_TEST_CASE(simple_mdpor_save_load_save_load) {
    RMDP rmdp1(0);

    rmdp1.add_transition_d(0,1,1,1,0);
    rmdp1.add_transition_d(1,1,2,1,0);
    rmdp1.add_transition_d(2,1,2,1,1.1);

    rmdp1.add_transition_d(0,0,0,1,0);
    rmdp1.add_transition_d(1,0,0,1,1);
    rmdp1.add_transition_d(2,0,1,1,1);

    Transition initial(indvec{0,1,2},numvec{1.0/3.0,1.0/3.0,1/3.0});
    indvec state2obs{0,0,1};

    MDPI_R mdpi1(rmdp1, state2obs, initial);

    stringstream store1, store2, store3;

    mdpi1.to_csv(store1,store2,store3);
    store1.seekg(0); store2.seekg(0); store3.seekg(0);

    auto&& string11 = store1.str();
    auto&& string12 = store2.str();
    auto&& string13 = store3.str();

    auto mdpi2 = MDPI_R::from_csv(store1,store2,store3);

    stringstream store21, store22, store23;

    mdpi2->to_csv(store21, store22, store23);

    auto&& string21 = store21.str();
    auto&& string22 = store22.str();
    auto&& string23 = store23.str();

    BOOST_CHECK_EQUAL(string11, string21);
    BOOST_CHECK_EQUAL(string12, string22);
    BOOST_CHECK_EQUAL(string13, string23);
}


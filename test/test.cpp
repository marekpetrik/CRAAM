#include "Transition.hpp"
#include "Action.hpp"
#include "State.hpp"
#include "RMDP.hpp"
#include "definitions.hpp"
#include "modeltools.hpp"

#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>

using namespace std;
using namespace craam;

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

// ********************************************************************************
// ***** Model construction methods ***********************************************
// ********************************************************************************
template<class Model>
Model create_test_mdp(){
    Model rmdp(3);

    // nonrobust and deterministic
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and rewards [0,0,1.1]
    add_transition<Model>(rmdp,0,1,1,1.0,0.0);
    add_transition<Model>(rmdp,1,1,2,1.0,0.0);
    add_transition<Model>(rmdp,2,1,2,1.0,1.1);

    add_transition<Model>(rmdp,0,0,0,1.0,0.0);
    add_transition<Model>(rmdp,1,0,0,1.0,1.0);
    add_transition<Model>(rmdp,2,0,1,1.0,1.0);

    return rmdp;
}


// ********************************************************************************
// ***** L1 worst case ************************************************************
// ********************************************************************************


BOOST_AUTO_TEST_CASE(test_l1_worst_case){
    numvec q = {0.4, 0.3, 0.1, 0.2};
    numvec z = {1.0, 2.0, 5.0, 4.0};
    prec_t t, w;

    t = 0;
    w = worstcase_l1(z,q,t).second;
    BOOST_CHECK_CLOSE(w, 2.3, 1e-3);

    t = 1;
    w = worstcase_l1(z,q,t).second;
    BOOST_CHECK_CLOSE(w,1.1,1e-3);

    t = 2;
    w = worstcase_l1(z,q,t).second;
    BOOST_CHECK_CLOSE(w,1,1e-3);

    numvec q1 = {1.0};
    numvec z1 = {2.0};

    t = 0;
    w = worstcase_l1(z1,q1,t).second;
    BOOST_CHECK_CLOSE(w,2.0,1e-3);

    t = 0.01;
    w = worstcase_l1(z1,q1,t).second;
    BOOST_CHECK_CLOSE(w, 2.0, 1e-3);

    t = 1;
    w = worstcase_l1(z1,q1,t).second;
    BOOST_CHECK_CLOSE(w, 2.0,1e-3);

    t = 2;
    w = worstcase_l1(z1,q1,t).second;
    BOOST_CHECK_CLOSE(w, 2.0,1e-3);
}


// ********************************************************************************
// ***** Basic solution tests **********************************************************
// ********************************************************************************

BOOST_AUTO_TEST_CASE( empty_test ){
    MDP m(0);
    m.mpi_jac(Uncertainty::Average, 0.9);
    m.vi_jac(Uncertainty::Average, 0.9);
    m.vi_gs(Uncertainty::Average, 0.9);
}

BOOST_AUTO_TEST_CASE( basic_tests ) {
    Transition t1({1,2}, {0.1,0.2}, {3,4});
    Transition t2({1,2}, {0.1,0.2}, {5,4});
    Transition t3({1,2}, {0.1,0.3}, {3,4});

    // check value computation
    numvec valuefunction = {0,1,2};
    auto ret = t1.compute_value(valuefunction,0.1);
    BOOST_CHECK (-0.01 <= ret - 1.15);
    BOOST_CHECK (ret - 1.15 <= 0.01);

    // check maximum selection
    DiscreteOutcomeAction a1({t1,t2}),a2({t1,t3});
    BOOST_CHECK_EQUAL(a1.maximal(valuefunction, 0.9).first, 1);
    BOOST_CHECK_EQUAL(a1.minimal(valuefunction,0.9).first, 0);
    BOOST_CHECK_EQUAL(a2.maximal(valuefunction, 0.9).first, 1);
    BOOST_CHECK_EQUAL(a2.minimal(valuefunction,0.9).first, 0);

    DiscreteOutcomeAction a3({t2});
    DiscreteRobustState s1({a1,a2,a3});
    auto v1 = get<2>(s1.max_max(valuefunction,0.9));
    auto v2 = get<2>(s1.max_min(valuefunction,0.9));
    BOOST_CHECK_LE (-0.01, v1-2.13);
    BOOST_CHECK_LE (v1-2.13, 0.01);
    BOOST_CHECK_LE (-0.01, v2-1.75);
    BOOST_CHECK_LE (v2-1.75, 0.01);
}

// ********************************************************************************
// ***** MDP value iteration ****************************************************
// ********************************************************************************

template<class Model>
void test_simple_vi(typename Model::OutcomePolicy natpol_rob){
    // Tests simple non-robust value iteration with the various models
    auto rmdp = create_test_mdp<Model>();

    Transition init_d({0,1,2},{1.0/3.0,1.0/3.0,1.0/3.0},{0,0,0});

    numvec initial{0,0,0};

    numvec val_rob{7.68072,8.67072,9.77072};
    indvec pol_rob{1,1,1};

    // small number of iterations (not the true value function)
    auto re = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,20,0);

    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // test jac value iteration
    auto re2 = rmdp.vi_jac(Uncertainty::Robust,0.9, initial,20,0);

    numvec val_rob2{7.5726,8.56265679,9.66265679};
    CHECK_CLOSE_COLLECTION(val_rob2,re2.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re2.policy.begin(),re2.policy.end());

    // many iterations
    const numvec val_rob3{8.91,9.9,11.0};
    const numvec occ_freq3{0.333333333,0.6333333333,9.03333333333333};
    const prec_t ret_true = 9.93666666;

    // robust
    auto&& re3 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re3.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re3.policy.begin(),re3.policy.end());

    auto&& re4 = rmdp.vi_jac(Uncertainty::Robust,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re4.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re4.policy.begin(),re4.policy.end());

    // optimistic
    auto&& re5 = rmdp.vi_gs(Uncertainty::Optimistic,0.9, initial, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re5.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re5.policy.begin(),re5.policy.end());

    auto&& re6 = rmdp.vi_jac(Uncertainty::Optimistic,0.9, initial, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re6.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re6.policy.begin(),re6.policy.end());

    // average
    auto&& re7 = rmdp.vi_gs(Uncertainty::Average, 0.9, initial, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re7.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re7.policy.begin(),re7.policy.end());

    auto&& re8 = rmdp.vi_jac(Uncertainty::Average,0.9, initial, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re8.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re8.policy.begin(),re8.policy.end());

    // fixed evaluation
    auto&& re9 = rmdp.vi_jac_fix(0.9,pol_rob,natpol_rob,initial,10000,0.0);
    CHECK_CLOSE_COLLECTION(val_rob3,re9.valuefunction,1e-2);

    // check the computed returns
    BOOST_CHECK_CLOSE (re3.total_return(init_d), ret_true, 1e-3);

    // check if we get the same return from the solution as from the
    // occupancy frequencies
    auto&& occupancy_freq = rmdp.ofreq_mat(init_d,0.9,re.policy,re.outcomes);
    CHECK_CLOSE_COLLECTION(occupancy_freq, occ_freq3, 1e-3);

    auto&& rewards = rmdp.rewards_state(re3.policy,re3.outcomes);
    auto cmp_tr = inner_product(rewards.begin(), rewards.end(), occupancy_freq.begin(), 0.0);
    BOOST_CHECK_CLOSE (cmp_tr, ret_true, 1e-3);
}

BOOST_AUTO_TEST_CASE(simple_mdp_vi_of_nonrobust) {
    test_simple_vi<MDP>(indvec{0,0,0});
}

BOOST_AUTO_TEST_CASE(simple_rmdpd_vi_of_nonrobust) {
    test_simple_vi<RMDP_D>(indvec{0,0,0});
}

BOOST_AUTO_TEST_CASE(simple_rmdpl1_vi_of_nonrobust) {
    test_simple_vi<RMDP_L1>(vector<numvec>{numvec{1},numvec{1},numvec{1}});
}


// ********************************************************************************
// ***** MDP modified policy iteration ********************************************
// ********************************************************************************

template<class Model>
void test_simple_mdp_mpi_like_vi(){

    // run mpi but use parameters that should recover the same solution as vi

    auto rmdp = create_test_mdp<Model>();

    numvec initial{0,0,0};

    // small number of iterations
    auto&& re = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,20,0);

    numvec val_rob{7.68072,8.67072,9.77072};
    indvec pol_rob{1,1,1};
    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    auto&& re2 = rmdp.mpi_jac(Uncertainty::Robust,0.9,initial,20,0,0,0);

    numvec val_rob2{7.5726,8.56265679,9.66265679};
    CHECK_CLOSE_COLLECTION(val_rob2,re2.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re2.policy.begin(),re2.policy.end());

    // many iterations
    numvec val_rob3{8.91,9.9,11.0};

    // robust
    auto&& re3 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re3.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re3.policy.begin(),re3.policy.end());

    auto&& re4 = rmdp.vi_jac(Uncertainty::Robust,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re4.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re4.policy.begin(),re4.policy.end());

    // optimistic
    auto&& re5 = rmdp.vi_gs(Uncertainty::Optimistic,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re5.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re5.policy.begin(),re5.policy.end());

    auto&& re6 = rmdp.vi_jac(Uncertainty::Optimistic,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re6.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re6.policy.begin(),re6.policy.end());

    // average
    auto&& re7 = rmdp.vi_gs(Uncertainty::Average,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re7.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re7.policy.begin(),re7.policy.end());

    auto&& re8 = rmdp.vi_jac(Uncertainty::Average,0.9,initial,10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re8.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re8.policy.begin(),re8.policy.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_mpi_like_vi_mdp) {
    test_simple_mdp_mpi_like_vi<MDP>();
}

BOOST_AUTO_TEST_CASE(simple_mdp_mpi_like_vi_drmdp) {
    test_simple_mdp_mpi_like_vi<RMDP_D>();
}

BOOST_AUTO_TEST_CASE(simple_mdp_mpi_like_vi_rmdpl1) {
    test_simple_mdp_mpi_like_vi<RMDP_L1>();
}


// ********************************************************************************
// ***** Model resize *************************************************************
// ********************************************************************************


template <class Model>
void test_check_add_transition(typename Model::OutcomeId firstoutcome){

    Model rmdp;

    // check adding to the end
    add_transition(rmdp,0,0,0,5,0.1,1);
    add_transition(rmdp,0,0,0,7,0.1,2);

    Transition&& transition = rmdp.get_state(0).mean_transition(0,firstoutcome);

    BOOST_CHECK(is_sorted(transition.get_indices().begin(), transition.get_indices().end()) );
    BOOST_CHECK_EQUAL(transition.get_indices().size(), 2);
    BOOST_CHECK_EQUAL(transition.get_probabilities().size(), 2);
    BOOST_CHECK_EQUAL(transition.get_rewards().size(), 2);

    // check updating the last element
    add_transition(rmdp,0,0,0,7,0.4,4);
    transition = rmdp.get_state(0).mean_transition(0,firstoutcome);

    BOOST_CHECK(is_sorted(transition.get_indices().begin(), transition.get_indices().end()) );
    BOOST_CHECK_EQUAL(transition.get_indices().size(), 2);
    BOOST_CHECK_EQUAL(transition.get_probabilities().size(), 2);
    BOOST_CHECK_EQUAL(transition.get_rewards().size(), 2);
    vector<double> tr{1.0,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_rewards().begin(), transition.get_rewards().end(), tr.begin(), tr.end());
    vector<double> tp{0.1,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_probabilities().begin(), transition.get_probabilities().end(), tp.begin(), tp.end());

    // check inserting an element into the middle
    add_transition(rmdp,0,0,0,6,0.1,0.5);
    transition = rmdp.get_state(0).mean_transition(0,firstoutcome);

    BOOST_CHECK(is_sorted(transition.get_indices().begin(), transition.get_indices().end()) );
    BOOST_CHECK_EQUAL(transition.get_indices().size(), 3);
    BOOST_CHECK_EQUAL(transition.get_probabilities().size(), 3);
    BOOST_CHECK_EQUAL(transition.get_rewards().size(), 3);
    tr = vector<double>{1.0,0.5,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_rewards().begin(), transition.get_rewards().end(), tr.begin(), tr.end());
    tp = vector<double>{0.1,0.1,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_probabilities().begin(), transition.get_probabilities().end(), tp.begin(), tp.end());

    // check updating an element in the middle
    add_transition(rmdp,0,0,0,6,0.1,1.5);
    transition = rmdp.get_state(0).mean_transition(0,firstoutcome);

    BOOST_CHECK(is_sorted(transition.get_indices().begin(), transition.get_indices().end()) );
    BOOST_CHECK_EQUAL(transition.get_indices().size(), 3);
    BOOST_CHECK_EQUAL(transition.get_probabilities().size(), 3);
    BOOST_CHECK_EQUAL(transition.get_rewards().size(), 3);
    tr = vector<double>{1.0,1.0,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_rewards().begin(), transition.get_rewards().end(), tr.begin(), tr.end());
    tp = vector<double>{0.1,0.2,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_probabilities().begin(), transition.get_probabilities().end(), tp.begin(), tp.end());
}

BOOST_AUTO_TEST_CASE(check_add_transition_mdp) {
    test_check_add_transition<MDP>(0);
}

BOOST_AUTO_TEST_CASE(check_add_transition_rmdpd) {
    test_check_add_transition<RMDP_D>(0);
}

BOOST_AUTO_TEST_CASE(check_add_transition_rmdpl1) {
    test_check_add_transition<RMDP_L1>(numvec{1.0});
}

// ********************************************************************************
// ***** Save and load ************************************************************
// ********************************************************************************

template<class Model>
void test_simple_mdp_save_load(){

    auto rmdp1 = create_test_mdp<Model>();

    stringstream store;

    rmdp1.to_csv(store);
    store.seekg(0);

    Model rmdp2;
    from_csv(rmdp2,store);

    numvec initial{0,0,0};

    auto&& re = rmdp2.vi_gs(Uncertainty::Robust,0.9,initial,20l,0);

    numvec val_rob{7.68072,8.67072,9.77072};
    indvec pol_rob{1,1,1};

    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load_mdp) {
    test_simple_mdp_save_load<MDP>();
}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load_rmdpd) {
    test_simple_mdp_save_load<RMDP_D>();
}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load_rmdpl1) {
    test_simple_mdp_save_load<RMDP_L1>();
}

template<class Model>
void test_simple_mdp_save_load_save_load() {
    Model rmdp1 = create_test_mdp<Model>();

    stringstream store;

    rmdp1.to_csv(store);
    store.seekg(0);

    auto string1 = store.str();

    Model rmdp2;
    from_csv(rmdp2,store);

    stringstream store2;

    rmdp2.to_csv(store2);

    auto string2 = store2.str();

    BOOST_CHECK_EQUAL(string1, string2);
}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load_save_load){
    test_simple_mdp_save_load_save_load<MDP>();
}

// ********************************************************************************
// ***** Value function ***********************************************************
// ********************************************************************************


template<class Model>
void test_value_function() {
    Model rmdp;

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);

    numvec initial{0};

    // gauss-seidel
    auto&& result1 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = rmdp.vi_gs(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    auto&& result3 = rmdp.vi_gs(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);

    // jacobi
    result1 = rmdp.vi_jac(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    result2 = rmdp.vi_jac(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    result3 = rmdp.vi_jac(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);
}


BOOST_AUTO_TEST_CASE(test_value_function_rmdpd){
    test_value_function<RMDP_D>();
}


// ********************************************************************************
// ***** L1 value function ********************************************************
// ********************************************************************************


template<class Model>
void test_value_function_thr() {
    Model rmdp;

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);
    numvec initial{0};

    numvec d{0.5,0.5};
    CHECK_CLOSE_COLLECTION(rmdp.get_state(0).get_action(0).get_distribution(), d, 1e-6);


    // *** 2.0 ***
    set_outcome_thresholds(rmdp, 2.0);
    BOOST_CHECK_CLOSE(rmdp.get_state(0).get_action(0).get_threshold(), 2.0, 1e-6);
    // gauss-seidel
    auto&& result1 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = rmdp.vi_gs(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    auto&& result3 = rmdp.vi_gs(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);

    // jacobi
    result1 = rmdp.vi_jac(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    result2 = rmdp.vi_jac(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    result3 = rmdp.vi_jac(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);

    // *** 1.0 ***
    set_outcome_thresholds(rmdp, 1.0);
    BOOST_CHECK_CLOSE(rmdp.get_state(0).get_action(0).get_threshold(), 1.0, 1e-6);
    // gauss-seidel
    result1 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    result2 = rmdp.vi_gs(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    result3 = rmdp.vi_gs(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);

    // *** 0.5 ***
    set_outcome_thresholds(rmdp, 0.5);
    BOOST_CHECK_CLOSE(rmdp.get_state(0).get_action(0).get_threshold(), 0.5, 1e-6);
    // gauss-seidel
    result1 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 12.5, 1e-3);

    result2 = rmdp.vi_gs(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 17.5, 1e-3);

    result3 = rmdp.vi_gs(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);

    // *** 0.0 ***
    set_outcome_thresholds(rmdp, 0.0);
    BOOST_CHECK_CLOSE(rmdp.get_state(0).get_action(0).get_threshold(), 0.0, 1e-6);
    // gauss-seidel
    result1 = rmdp.vi_gs(Uncertainty::Robust,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 15.0, 1e-3);

    result2 = rmdp.vi_gs(Uncertainty::Optimistic,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 15.0, 1e-3);

    result3 = rmdp.vi_gs(Uncertainty::Average,0.9,initial,1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);
}


BOOST_AUTO_TEST_CASE(test_value_function_rmdpl1){
    test_value_function_thr<RMDP_L1>();
}


// ********************************************************************************
// ***** String output ************************************************************
// ********************************************************************************



BOOST_AUTO_TEST_CASE(test_string_mdp){
    MDP rmdp;
    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,1,0,0,0,1,1);

    auto s = rmdp.to_string();
    BOOST_CHECK_EQUAL(s.length(), 42);
}

BOOST_AUTO_TEST_CASE(test_string_rmdpd){
    RMDP_D rmdp;

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);

    add_transition(rmdp,1,0,0,0,1,1);
    add_transition(rmdp,1,0,1,0,1,2);

    auto s = rmdp.to_string();
    BOOST_CHECK_EQUAL(s.length(), 32);
}


BOOST_AUTO_TEST_CASE(test_string_rmdpl1){
    RMDP_L1 rmdp;

    numvec dist{0.5,0.5};

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);
    //rmdp.get_state(0).get_action(0).set_distribution(dist);
    //rmdp.get_state(0).get_action(0).set_threshold(2);

    add_transition(rmdp,1,0,0,0,1,1);
    add_transition(rmdp,1,0,1,0,1,2);
    //rmdp.get_state(1).get_action(0).set_distribution(dist);
    //rmdp.get_state(1).get_action(0).set_threshold(2);

    set_outcome_thresholds(rmdp, 2);
    set_uniform_outcome_dst(rmdp);

    auto s = rmdp.to_string();
    BOOST_CHECK_EQUAL(s.length(), 40);
}

// ********************************************************************************
// ***** Normalization ************************************************************
// ********************************************************************************


BOOST_AUTO_TEST_CASE(test_normalization) {
    RMDP_L1 rmdp;

    // nonrobust
    add_transition(rmdp,0,0,0,1.0,0.1);
    add_transition(rmdp,0,0,1,1.0,0.5);

    // the freshly constructed one should be normalized
    BOOST_CHECK(is_outcome_dst_normalized(rmdp));

    // denormalize and make sure it works
    rmdp.get_state(0).get_action(0).set_distribution(0, 0.8);
    BOOST_CHECK(!is_outcome_dst_normalized(rmdp));

    // make sure that the normalization works
    normalize_outcome_dst(rmdp);
    BOOST_CHECK(is_outcome_dst_normalized(rmdp));

    // check and normalize transition probabilities
    BOOST_CHECK(!rmdp.is_normalized());
    rmdp.normalize();
    BOOST_CHECK(rmdp.is_normalized());

    // solve and check value function
    set_outcome_thresholds(rmdp, 2.0);
    numvec initial{0,0};
    auto&& re = rmdp.vi_jac(Uncertainty::Robust,0.9,initial,2000,0);

    numvec val{0.545454545455, 0.0};

    CHECK_CLOSE_COLLECTION(val,re.valuefunction,1e-3);
}

// ********************************************************************************
// ***** Stochastic transition probabilities (L1) *********************************
// ********************************************************************************


BOOST_AUTO_TEST_CASE(test_randomized_mdp){
    RMDP_L1 rmdp;

    // define the MDP representation
    // format: idstatefrom, idaction, idoutcome, idstateto, probability, reward
    string string_representation{
        "1,0,0,1,1.0,2.0 \
         2,0,0,2,1.0,3.0 \
         3,0,0,3,1.0,1.0 \
         4,0,0,4,1.0,4.0 \
         0,0,0,1,1.0,0.0 \
         0,0,1,2,1.0,0.0 \
         0,1,0,3,1.0,0.0 \
         0,1,1,4,1.0,0.0\n"};

    // initialize desired outcomes
    numvec robust_2_0 {0.9*20,20,30,10,40};
    numvec robust_1_0 {0.9*20,20,30,10,40};
    numvec robust_0_5 {0.9*90.0/4.0,20,30,10,40};
    numvec robust_0_0 {0.9*25.0,20,30,10,40};
    numvec optimistic_2_0 {0.9*40,20,30,10,40};
    numvec optimistic_1_0 {0.9*40,20,30,10,40};
    numvec optimistic_0_5 {0.9*130.0/4.0,20,30,10,40};
    numvec optimistic_0_0 {0.9*25.0,20,30,10,40};

    stringstream store(string_representation);

    store.seekg(0);
    from_csv(rmdp,store,false);

    // print the problem definition for debugging
    //cout << string_representation << endl;
    //cout << rmdp->state_count() << endl;
    //stringstream store2;
    //rmdp->to_csv(store2);
    //cout << store2.str() << endl;

    numvec value(5,0.0);
    const prec_t gamma = 0.9;

    // *** ROBUST ******************

    set_outcome_thresholds(rmdp,2.0);
    auto&& sol1 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol1.valuefunction, robust_2_0, 0.001);
    auto&& sol2 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol2.valuefunction, robust_2_0, 0.001);
    auto&& sol3 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol3.valuefunction, robust_2_0, 0.001);

    // should be the same without the l1 bound
    //set_outcome_thresholds(rmdp,2.0);
    //auto&& sol4 = rmdp.vi_jac_rob(value,gamma,1000,1e-5);
    //CHECK_CLOSE_COLLECTION(sol4.valuefunction, robust_2_0, 0.001);
    //auto&& sol5 = rmdp.vi_gs_rob(value,gamma,1000,1e-5);
    //CHECK_CLOSE_COLLECTION(sol5.valuefunction, robust_2_0, 0.001);
    //auto&& sol6 = rmdp.mpi_jac_rob(value,gamma,1000,1e-5,1000,1e-5);
    //CHECK_CLOSE_COLLECTION(sol6.valuefunction, robust_2_0, 0.001);

    set_outcome_thresholds(rmdp,1.0);
    auto&& sol7 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol7.valuefunction, robust_1_0, 0.001);
    auto&& sol8 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol8.valuefunction, robust_1_0, 0.001);
    auto&& sol9 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol9.valuefunction, robust_1_0, 0.001);

    set_outcome_thresholds(rmdp,0.5);
    auto&& sol10 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol10.valuefunction, robust_0_5, 0.001);
    auto&& sol11 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol11.valuefunction, robust_0_5, 0.001);
    auto&& sol12 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol12.valuefunction, robust_0_5, 0.001);

    set_outcome_thresholds(rmdp,0.0);
    auto&& sol13 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol13.valuefunction, robust_0_0, 0.001);
    auto&& sol14 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol14.valuefunction, robust_0_0, 0.001);
    auto&& sol15 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol15.valuefunction, robust_0_0, 0.001);

    // should be the same for the average
    set_outcome_thresholds(rmdp,0.0);
    auto&& sol16 = rmdp.vi_jac(Uncertainty::Average,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol16.valuefunction, robust_0_0, 0.001);
    auto&& sol17 = rmdp.vi_gs(Uncertainty::Average,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol17.valuefunction, robust_0_0, 0.001);
    auto&& sol18 = rmdp.mpi_jac(Uncertainty::Average,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol18.valuefunction, robust_0_0, 0.001);


    // *** OPTIMISTIC ******************

    set_outcome_thresholds(rmdp,2.0);
    auto&& sol19 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol19.valuefunction, optimistic_2_0, 0.001);
    auto&& sol20 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol20.valuefunction, optimistic_2_0, 0.001);
    auto&& sol21 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol21.valuefunction, optimistic_2_0, 0.001);

    // should be the same without the l1 bound
    set_outcome_thresholds(rmdp,2.0);
    auto&& sol22 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol22.valuefunction, optimistic_2_0, 0.001);
    auto&& sol23 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol23.valuefunction, optimistic_2_0, 0.001);
    auto&& sol24 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol24.valuefunction, optimistic_2_0, 0.001);

    set_outcome_thresholds(rmdp,1.0);
    auto&& sol25 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol25.valuefunction, optimistic_1_0, 0.001);
    auto&& sol26 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol26.valuefunction, optimistic_1_0, 0.001);
    auto&& sol27 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol27.valuefunction, optimistic_1_0, 0.001);

    set_outcome_thresholds(rmdp,0.5);
    auto&& sol28 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol28.valuefunction, optimistic_0_5, 0.001);
    auto&& sol29 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol29.valuefunction, optimistic_0_5, 0.001);
    auto&& sol30 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol30.valuefunction, optimistic_0_5, 0.001);

    set_outcome_thresholds(rmdp,0.0);
    auto&& sol31 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol31.valuefunction, optimistic_0_0, 0.001);
    auto&& sol32 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol32.valuefunction, optimistic_0_0, 0.001);
    auto&& sol33 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol33.valuefunction, optimistic_0_0, 0.001);
}


// ********************************************************************************
// ***** Test terminal state ******************************************************
// ********************************************************************************

BOOST_AUTO_TEST_CASE(test_randomized_mdp_with_terminal_state){
    RMDP_L1 rmdp;

    // define the MDP representation
    // format: idstatefrom, idaction, idoutcome, idstateto, probability, reward
    string string_representation{
        "1,0,0,5,1.0,20.0 \
         2,0,0,5,1.0,30.0 \
         3,0,0,5,1.0,10.0 \
         4,0,0,5,1.0,40.0 \
         0,0,0,1,1.0,0.0 \
         0,0,1,2,1.0,0.0 \
         0,1,0,3,1.0,0.0 \
         0,1,1,4,1.0,0.0\n"};

    // the last state is terminal

    // initialize desired outcomes
    numvec robust_2_0 {0.9*20,20,30,10,40,0};
    numvec robust_1_0 {0.9*20,20,30,10,40,0};
    numvec robust_0_5 {0.9*90.0/4.0,20,30,10,40,0};
    numvec robust_0_0 {0.9*25.0,20,30,10,40,0};
    numvec optimistic_2_0 {0.9*40,20,30,10,40,0};
    numvec optimistic_1_0 {0.9*40,20,30,10,40,0};
    numvec optimistic_0_5 {0.9*130.0/4.0,20,30,10,40,0};
    numvec optimistic_0_0 {0.9*25.0,20,30,10,40,0};

    stringstream store(string_representation);

    store.seekg(0);
    from_csv(rmdp,store,false);

    // print the problem definition for debugging
    //cout << string_representation << endl;
    //cout << rmdp->state_count() << endl;
    //stringstream store2;
    //rmdp->to_csv(store2);
    //cout << store2.str() << endl;

    numvec value(6,0.0);
    const prec_t gamma = 0.9;

    // *** ROBUST ******************

    set_outcome_thresholds(rmdp,2.0);
    auto&& sol1 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol1.valuefunction, robust_2_0, 0.001);
    auto&& sol2 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol2.valuefunction, robust_2_0, 0.001);
    auto&& sol3 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol3.valuefunction, robust_2_0, 0.001);

    // should be the same without the l1 bound
    //set_outcome_thresholds(rmdp,2.0);
    //auto&& sol4 = rmdp.vi_jac_rob(value,gamma,1000,1e-5);
    //CHECK_CLOSE_COLLECTION(sol4.valuefunction, robust_2_0, 0.001);
    //auto&& sol5 = rmdp.vi_gs_rob(value,gamma,1000,1e-5);
    //CHECK_CLOSE_COLLECTION(sol5.valuefunction, robust_2_0, 0.001);
    //auto&& sol6 = rmdp.mpi_jac_rob(value,gamma,1000,1e-5,1000,1e-5);
    //CHECK_CLOSE_COLLECTION(sol6.valuefunction, robust_2_0, 0.001);

    set_outcome_thresholds(rmdp,1.0);
    auto&& sol7 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol7.valuefunction, robust_1_0, 0.001);
    auto&& sol8 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol8.valuefunction, robust_1_0, 0.001);
    auto&& sol9 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol9.valuefunction, robust_1_0, 0.001);

    set_outcome_thresholds(rmdp,0.5);
    auto&& sol10 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol10.valuefunction, robust_0_5, 0.001);
    auto&& sol11 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol11.valuefunction, robust_0_5, 0.001);
    auto&& sol12 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol12.valuefunction, robust_0_5, 0.001);

    set_outcome_thresholds(rmdp,0.0);
    auto&& sol13 = rmdp.vi_jac(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol13.valuefunction, robust_0_0, 0.001);
    auto&& sol14 = rmdp.vi_gs(Uncertainty::Robust,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol14.valuefunction, robust_0_0, 0.001);
    auto&& sol15 = rmdp.mpi_jac(Uncertainty::Robust,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol15.valuefunction, robust_0_0, 0.001);

    // should be the same for the average
    set_outcome_thresholds(rmdp,0.0);
    auto&& sol16 = rmdp.vi_jac(Uncertainty::Average,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol16.valuefunction, robust_0_0, 0.001);
    auto&& sol17 = rmdp.vi_gs(Uncertainty::Average,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol17.valuefunction, robust_0_0, 0.001);
    auto&& sol18 = rmdp.mpi_jac(Uncertainty::Average,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol18.valuefunction, robust_0_0, 0.001);


    // *** OPTIMISTIC ******************

    set_outcome_thresholds(rmdp,2.0);
    auto&& sol19 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol19.valuefunction, optimistic_2_0, 0.001);
    auto&& sol20 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol20.valuefunction, optimistic_2_0, 0.001);
    auto&& sol21 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol21.valuefunction, optimistic_2_0, 0.001);

    // should be the same without the l1 bound
    set_outcome_thresholds(rmdp,2.0);
    auto&& sol22 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol22.valuefunction, optimistic_2_0, 0.001);
    auto&& sol23 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol23.valuefunction, optimistic_2_0, 0.001);
    auto&& sol24 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol24.valuefunction, optimistic_2_0, 0.001);

    set_outcome_thresholds(rmdp,1.0);
    auto&& sol25 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol25.valuefunction, optimistic_1_0, 0.001);
    auto&& sol26 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol26.valuefunction, optimistic_1_0, 0.001);
    auto&& sol27 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol27.valuefunction, optimistic_1_0, 0.001);

    set_outcome_thresholds(rmdp,0.5);
    auto&& sol28 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol28.valuefunction, optimistic_0_5, 0.001);
    auto&& sol29 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol29.valuefunction, optimistic_0_5, 0.001);
    auto&& sol30 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol30.valuefunction, optimistic_0_5, 0.001);

    set_outcome_thresholds(rmdp,0.0);
    auto&& sol31 = rmdp.vi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol31.valuefunction, optimistic_0_0, 0.001);
    auto&& sol32 = rmdp.vi_gs(Uncertainty::Optimistic,gamma,value,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol32.valuefunction, optimistic_0_0, 0.001);
    auto&& sol33 = rmdp.mpi_jac(Uncertainty::Optimistic,gamma,value,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol33.valuefunction, optimistic_0_0, 0.001);

}

// ********************************************************************************
//          Test adding outcomes to a weighted action
// ********************************************************************************

BOOST_AUTO_TEST_CASE(test_create_outcome){

    L1OutcomeAction a;
    numvec desired(5,0.2);  // this is the correct distribution with 5 outcomes

    a.create_outcome(1);
    //cout << a.get_distribution() << endl;
    a.create_outcome(2);
    //cout << a.get_distribution() << endl;
    a.create_outcome(0);
    //cout << a.get_distribution() << endl;
    a.create_outcome(4);

    auto d1 = a.get_distribution();
    //cout << d1 << endl;
    CHECK_CLOSE_COLLECTION(d1, desired, 0.0001);
    BOOST_CHECK(a.is_distribution_normalized());

    a.normalize_distribution();

    // make sure that normalization works too
    auto d2 = a.get_distribution();
    CHECK_CLOSE_COLLECTION(d2, desired, 0.0001);
    BOOST_CHECK(a.is_distribution_normalized());

}

// ********************************************************************************
// ***** Test CSV *****************************************************************
// ********************************************************************************


BOOST_AUTO_TEST_CASE(test_parameter_read_write){

    RMDP_L1 rmdp;

    // define the MDP representation
    // format: idstatefrom, idaction, idoutcome, idstateto, probability, reward
    string string_representation{
        "1,0,0,5,1.0,20.0 \
         2,0,0,5,1.0,30.0 \
         3,0,0,5,1.0,10.0 \
         4,0,0,5,1.0,40.0 \
         4,1,0,5,1.0,41.0 \
         0,0,0,1,1.0,0.0 \
         0,0,1,2,1.0,0.0 \
         0,1,0,3,1.0,0.0 \
         0,1,0,4,1.0,2.0 \
         0,1,1,4,1.0,0.0\n"};

    stringstream store(string_representation);

    store.seekg(0);
    from_csv(rmdp,store,false);

    BOOST_CHECK_EQUAL(rmdp.get_state(3).get_action(0).get_outcome(0).get_reward(0), 10.0);
    rmdp.get_state(3).get_action(0).get_outcome(0).set_reward(0,15.1);
    BOOST_CHECK_EQUAL(rmdp.get_state(3).get_action(0).get_outcome(0).get_reward(0), 15.1);

    BOOST_CHECK_EQUAL(rmdp.get_state(0).get_action(1).get_outcome(0).get_reward(1), 2.0);
    rmdp.get_state(0).get_action(1).get_outcome(0).set_reward(1,19.1);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).get_action(1).get_outcome(0).get_reward(1), 19.1);

    rmdp.get_state(3).get_action(0).set_threshold(1.0);
    BOOST_CHECK_EQUAL(rmdp.get_state(3).get_action(0).get_threshold(), 1.0);

}

// ********************************************************************************
//  Test robustification 
// ********************************************************************************

BOOST_AUTO_TEST_CASE(test_robustification){
    MDP mdp = create_test_mdp<MDP>();
    
    // no transition to zero probability states
    RMDP_L1 rmdp_nz = robustify<L1RobustState>(mdp, false);
    // allow transitions to zero probability states
    RMDP_L1 rmdp_z = robustify<L1RobustState>(mdp, true);
    

    
    cout << "MDP Solution " << mdp.mpi_jac(Uncertainty::Robust, 0.9).valuefunction << endl;
    cout << "RMDP Solution, no zeros " << rmdp_nz.mpi_jac(Uncertainty::Robust, 0.9).valuefunction << endl;
    cout << "MDP Solution, zeros " << rmdp_z.mpi_jac(Uncertainty::Robust, 0.9).valuefunction << endl;
    
    set_outcome_thresholds(rmdp_nz, 2.0);
    set_outcome_thresholds(rmdp_z, 2.0);

    cout << "RMDP Solution, no zeros " << rmdp_nz.mpi_jac(Uncertainty::Robust, 0.9).valuefunction << endl;
    cout << "MDP Solution, zeros " << rmdp_z.mpi_jac(Uncertainty::Robust, 0.9).valuefunction << endl;

    cout << rmdp_z.to_json() << endl;
}

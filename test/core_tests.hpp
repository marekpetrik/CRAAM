// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "craam/definitions.hpp"
#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"
#include "craam/algorithms/robust_values.hpp"
#include "craam/algorithms/occupancies.hpp"
#include "craam/optimization/bisection.hpp"
#include "craam/algorithms/nature_response.hpp"
#include "craam/optimization/srect_gurobi.hpp"

#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <utility>

using namespace std;
using namespace craam;
using namespace craam::algorithms;

// ********************************************************************************
// ***** Model construction methods ***********************************************
// ********************************************************************************
template<class Model>
Model create_test_mdp(){
    Model rmdp(3);

    // nonrobust and deterministic
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and rewards [0,0,1.1]
    // action 0 has a transition matrix [[1,0,0],[1,0,0], [0,1,0]] and rewards [0,1.0,1.0]
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

    vi_gs(m, 0.9);
    mpi_jac(m, 0.9);
}

BOOST_AUTO_TEST_CASE( basic_tests ) {
    Transition t1({1,2}, {0.1,0.2}, {3,4});
    Transition t2({1,2}, {0.1,0.2}, {5,4});
    Transition t3({1,2}, {0.1,0.3}, {3,4});

    // check value computation
    numvec valuefunction = {0,1,2};
    auto ret = t1.value(valuefunction,0.1);
    BOOST_CHECK_CLOSE ( ret, 1.15, 1e-3);

    // check values of transitions:
    BOOST_CHECK_CLOSE(t1.value(valuefunction, 0.9), 0.1*(3 + 0.9*1) + 0.2*(4 + 0.9*2), 1e-3);
    BOOST_CHECK_CLOSE(t2.value(valuefunction, 0.9), 0.1*(5 + 0.9*1) + 0.2*(4 + 0.9*2), 1e-3);
    BOOST_CHECK_CLOSE(t3.value(valuefunction, 0.9), 0.1*(3 + 0.9*1) + 0.3*(4 + 0.9*2), 1e-3);

    // check values of actions
    WeightedOutcomeAction a1({t1,t2}),a2({t1,t3});
    WeightedOutcomeAction a3({t2});

    BOOST_CHECK_CLOSE(value_action(a1, valuefunction, 0.9), 0.5*(t1.value(valuefunction, 0.9)+
                                                                 t2.value(valuefunction, 0.9)), 1e-3);
    BOOST_CHECK_CLOSE(value_action(a1, valuefunction, 0.9, 0,0, nats::robust_unbounded()).second,
                                min(t1.value(valuefunction, 0.9), t2.value(valuefunction, 0.9)), 1e-3);
    BOOST_CHECK_CLOSE(value_action(a1, valuefunction, 0.9, 0,0, nats::optimistic_unbounded()).second,
                                max(t1.value(valuefunction, 0.9), t2.value(valuefunction, 0.9)), 1e-3);

    WeightedRobustState s1({a1,a2,a3});
    auto v1 = get<2>(value_max_state(s1,valuefunction,0.9,0,nats::optimistic_unbounded()));
    auto v2 = get<2>(value_max_state(s1,valuefunction,0.9,0,nats::robust_unbounded()));
    BOOST_CHECK_CLOSE (v1, 2.13, 1e-3);
    BOOST_CHECK_CLOSE (v2, 1.75, 1e-3);
}

// ********************************************************************************
// ***** MDP value iteration ****************************************************
// ********************************************************************************

template<class Model>
void test_simple_vi(const Model& rmdp){
    // Tests simple non-robust value iteration with the various models

    indvec natpol_rob{0,0,0};
    Transition init_d({0,1,2},{1.0/3.0,1.0/3.0,1.0/3.0},{0,0,0});

    numvec initial{0,0,0};
    indvec pol_rob{1,1,1};

    // small number of iterations (not the true value function)
    numvec val_rob{7.68072,8.67072,9.77072};
    auto re = vi_gs(rmdp,0.9,initial,PlainBellman<typename Model::state_type>(indvec(0)),20,0);

    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // test jac value iteration with small number of iterations ( not the true value function)
    auto re2 = mpi_jac(rmdp, 0.9, initial, PlainBellman<typename Model::state_type>(indvec(0)), 20,0,0);

    numvec val_rob2{7.5726,8.56265679,9.66265679};
    CHECK_CLOSE_COLLECTION(val_rob2,re2.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re2.policy.begin(),re2.policy.end());

    // many iterations
    const numvec val_rob3{8.91,9.9,11.0};
    const numvec occ_freq3{0.333333333,0.6333333333,9.03333333333333};
    const prec_t ret_true = inner_product(val_rob3.cbegin(), val_rob3.cend(), init_d.get_probabilities().cbegin(),0.0);

    // robust
    auto&& re3 = vi_gs(rmdp,0.9,initial, SARobustBellman<typename Model::state_type>(nats::robust_l1u(0.0)));
    CHECK_CLOSE_COLLECTION(val_rob3,re3.valuefunction,1e-2);
    auto re3_pol = unzip(re3.policy).first;
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re3_pol.begin(),re3_pol.end());

    auto&& re4 = mpi_jac(rmdp,0.9, initial, SARobustBellman<typename Model::state_type>(nats::robust_l1u(0.0)),
                            1000, 0.0, 1000, 0.0);
    CHECK_CLOSE_COLLECTION(val_rob3,re4.valuefunction,1e-2);
    auto re4_pol = unzip(re4.policy).first;
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re4_pol.begin(),re4_pol.end());

    // optimistic
    auto&& re5 = vi_gs(rmdp,0.9, initial, SARobustBellman<typename Model::state_type>(nats::optimistic_l1u(0.0)));
    CHECK_CLOSE_COLLECTION(val_rob3,re5.valuefunction,1e-2);
    auto re5_pol = unzip(re5.policy).first;
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re5_pol.begin(),re5_pol.end());

    auto&& re6 = mpi_jac(rmdp,0.9, initial, SARobustBellman<typename Model::state_type>(nats::optimistic_l1u(0.0)));
    CHECK_CLOSE_COLLECTION(val_rob3,re6.valuefunction,1e-2);
    auto re6_pol = unzip(re6.policy).first;
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re6_pol.begin(),re6_pol.end());

    // plain
    auto&& re7 = vi_gs(rmdp, 0.9, initial);
    CHECK_CLOSE_COLLECTION(val_rob3,re7.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re7.policy.begin(),re7.policy.end());

    auto&& re8 = mpi_jac(rmdp, 0.9, initial, PlainBellman<typename Model::state_type>());
    CHECK_CLOSE_COLLECTION(val_rob3,re8.valuefunction,1e-2);
    auto re8_pol = re8.policy;
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re8_pol.begin(),re8_pol.end());

    // fixed evaluation
    auto&& re9 = mpi_jac(rmdp,0.9,initial,PlainBellman<typename Model::state_type>(pol_rob), 10000,0.0, 0);
    CHECK_CLOSE_COLLECTION(val_rob3,re9.valuefunction,1e-2);

    // check the computed returns
    BOOST_CHECK_CLOSE (re8.total_return(init_d), ret_true, 1e-2);

    // check if we get the same return from the solution as from the
    // occupancy frequencies
    auto&& occupancy_freq = occfreq_mat(rmdp, init_d,0.9,re.policy);
    CHECK_CLOSE_COLLECTION(occupancy_freq, occ_freq3, 1e-3);

    auto&& rewards = rewards_vec(rmdp, re3_pol);
    auto cmp_tr = inner_product(rewards.begin(), rewards.end(), occupancy_freq.begin(), 0.0);
    BOOST_CHECK_CLOSE (cmp_tr, ret_true, 1e-3);
}

BOOST_AUTO_TEST_CASE(simple_mdp_vi_of_nonrobust) {
    auto rmdp = create_test_mdp<MDP>();
    test_simple_vi<MDP>(rmdp);
}

BOOST_AUTO_TEST_CASE(simple_rmdpd_vi_of_nonrobust) {
    auto rmdp = create_test_mdp<MDP>();
    test_simple_vi<RMDP>(robustify(rmdp));
}

// ********************************************************************************
// ***** Model resize *************************************************************
// ********************************************************************************


BOOST_AUTO_TEST_CASE(test_check_add_transition_m){

    MDP rmdp;

    // check adding to the end
    add_transition(rmdp,0,0,0,5,0.1,1);
    add_transition(rmdp,0,0,0,7,0.1,2);

    Transition&& transition = rmdp.get_state(0).mean_transition(0);

    BOOST_CHECK(is_sorted(transition.get_indices().begin(), transition.get_indices().end()) );
    BOOST_CHECK_EQUAL(transition.get_indices().size(), 2);
    BOOST_CHECK_EQUAL(transition.get_probabilities().size(), 2);
    BOOST_CHECK_EQUAL(transition.get_rewards().size(), 2);

    // check updating the last element
    add_transition(rmdp,0,0,0,7,0.4,4);
    transition = rmdp.get_state(0).mean_transition(0);

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
    transition = rmdp.get_state(0).mean_transition(0);

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
    transition = rmdp.get_state(0).mean_transition(0);

    BOOST_CHECK(is_sorted(transition.get_indices().begin(), transition.get_indices().end()) );
    BOOST_CHECK_EQUAL(transition.get_indices().size(), 3);
    BOOST_CHECK_EQUAL(transition.get_probabilities().size(), 3);
    BOOST_CHECK_EQUAL(transition.get_rewards().size(), 3);
    tr = vector<double>{1.0,1.0,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_rewards().begin(), transition.get_rewards().end(), tr.begin(), tr.end());
    tp = vector<double>{0.1,0.2,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.get_probabilities().begin(), transition.get_probabilities().end(), tp.begin(), tp.end());
}


BOOST_AUTO_TEST_CASE(test_check_add_transition_r){

    RMDP rmdp;

    numvec firstoutcome = numvec{1.0};
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

// ********************************************************************************
// ***** Save and load ************************************************************
// ********************************************************************************

template<class Model>
void test_simple_mdp_save_load(){

    auto rmdp1 = create_test_mdp<Model>();

    stringstream store;

    to_csv(rmdp1, store);
    store.seekg(0);

    Model rmdp2;
    from_csv(rmdp2,store);

    numvec initial{0,0,0};

    auto&& re = vi_gs(rmdp2,0.9,initial,SARobustBellman<typename Model::state_type>(nats::robust_l1u(0.0)),20l,0);

    numvec val_rob{7.68072,8.67072,9.77072};
    indvec pol_rob{1,1,1};

    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    auto re_policy = unzip(re.policy).first;
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re_policy.begin(),re_policy.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load_mdp) {
    test_simple_mdp_save_load<MDP>();
}
 
BOOST_AUTO_TEST_CASE(simple_mdp_save_load_rmdpd) {
    test_simple_mdp_save_load<RMDP>();
}


template<class Model>
void test_simple_mdp_save_load_save_load() {
    Model rmdp1 = create_test_mdp<Model>();

    stringstream store;

    to_csv(rmdp1, store);
    store.seekg(0);

    auto string1 = store.str();

    Model rmdp2;
    from_csv(rmdp2,store);

    stringstream store2;

    to_csv(rmdp2, store2);

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
void test_value_function(const Model& rmdp) {
    numvec initial{0};

    // gauss-seidel
    auto&& result1 = vi_gs(rmdp,0.9,initial,SARobustBellman<typename Model::state_type>(nats::robust_unbounded()), 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = vi_gs(rmdp,0.9,initial,SARobustBellman<typename Model::state_type>(nats::optimistic_unbounded()), 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    auto&& result3 = vi_gs(rmdp,0.9,initial,PlainBellman<typename Model::state_type>(), 1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);

    // mpi
    result1 = mpi_jac(rmdp,0.9,initial,SARobustBellman<typename Model::state_type>(nats::robust_unbounded()), 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    result2 = mpi_jac(rmdp,0.9,initial,SARobustBellman<typename Model::state_type>(nats::optimistic_unbounded()), 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    result3 = mpi_jac(rmdp,0.9,initial,PlainBellman<typename Model::state_type>(), 1000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);
}


BOOST_AUTO_TEST_CASE(test_value_function_rmdp){
    RMDP rmdp;

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);
    test_value_function<RMDP>(rmdp);
}

// ********************************************************************************
// ***** L1 value function ********************************************************
// ********************************************************************************


void test_value_function_thr(double threshold, numvec expected) {
    RMDP rmdp;

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);
    numvec initial{0};

    numvec d{0.5,0.5};
    CHECK_CLOSE_COLLECTION(rmdp.get_state(0).get_action(0).get_distribution(), d, 1e-6);


    // *** 2.0 ***
    // gauss-seidel
    auto&& result1 = vi_gs(rmdp,0.9,initial,SARobustBellman<WeightedRobustState>(nats::robust_l1u(threshold)), 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], expected[0], 1e-3);

    auto&& result2 = vi_gs(rmdp,0.9,initial,SARobustBellman<WeightedRobustState>(nats::optimistic_l1u(threshold)), 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], expected[1], 1e-3);

    // mpi
    result1 = mpi_jac(rmdp,0.9,initial,SARobustBellman<WeightedRobustState>(nats::robust_l1u(threshold)), 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], expected[0], 1e-3);

    result2 = mpi_jac(rmdp,0.9,initial,SARobustBellman<WeightedRobustState>(nats::optimistic_l1u(threshold)), 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], expected[1], 1e-3);
}


BOOST_AUTO_TEST_CASE(test_value_function_rmdpl1){
    test_value_function_thr(2.0, numvec{10.0,20.0});
    test_value_function_thr(1.0, numvec{10.0, 20.0});
    test_value_function_thr(0.5, numvec{12.5, 17.5});
    test_value_function_thr(0.0, numvec{15.0, 15.0});
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



BOOST_AUTO_TEST_CASE(test_string_rmdpl1){
    RMDP rmdp;

    numvec dist{0.5,0.5};

    add_transition(rmdp,0,0,0,0,1,1);
    add_transition(rmdp,0,0,1,0,1,2);

    add_transition(rmdp,1,0,0,0,1,1);
    add_transition(rmdp,1,0,1,0,1,2);

    set_uniform_outcome_dst(rmdp);

    auto s = rmdp.to_string();
    BOOST_CHECK_EQUAL(s.length(), 40);
}

// ********************************************************************************
// ***** Normalization ************************************************************
// ********************************************************************************


BOOST_AUTO_TEST_CASE(test_normalization) {
    RMDP rmdp;

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

    // check and normalize outcome probabilities
    BOOST_CHECK(!rmdp.is_normalized());
    rmdp.normalize();
    BOOST_CHECK(rmdp.is_normalized());

    // solve and check value function
    numvec initial{0,0};
    auto&& re = mpi_jac(rmdp,0.9,initial,SARobustBellman<WeightedRobustState>(nats::robust_unbounded()), 2000,0);

    numvec val{0.545454545455, 0.0};

    CHECK_CLOSE_COLLECTION(val,re.valuefunction,1e-3);
}

// ********************************************************************************
// ***** Stochastic transition probabilities (L1) *********************************
// ********************************************************************************

void test_randomized_threshold_average(const RMDP& rmdp, const numvec& desired){

    const prec_t gamma = 0.9;
    numvec value(0);
    auto&& sol2 = vi_gs(rmdp,gamma,value,PlainBellman<WeightedRobustState>(),1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol2.valuefunction, desired, 0.001);
    auto&& sol3 = mpi_jac(rmdp,gamma,value,PlainBellman<WeightedRobustState>(),1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol3.valuefunction, desired, 0.001);
}



void test_randomized_threshold_robust(const RMDP& rmdp, double threshold, const numvec& desired){

    const prec_t gamma = 0.9;
    numvec value(0);
    auto&& sol2 = vi_gs(rmdp,gamma,value,SARobustBellman<WeightedRobustState>(nats::robust_l1u(threshold)),1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol2.valuefunction, desired, 0.001);
    auto&& sol3 = mpi_jac(rmdp,gamma,value,SARobustBellman<WeightedRobustState>(nats::robust_l1u(threshold)),1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol3.valuefunction, desired, 0.001);

}

void test_randomized_threshold_optimistic(const RMDP& rmdp, double threshold, const numvec& desired){

    const prec_t gamma = 0.9;
    numvec value(0);
    auto&& sol2 = vi_gs(rmdp,gamma,value,SARobustBellman<WeightedRobustState>(nats::optimistic_l1u(threshold)),1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol2.valuefunction, desired, 0.001);
    auto&& sol3 = mpi_jac(rmdp,gamma,value,SARobustBellman<WeightedRobustState>(nats::optimistic_l1u(threshold)),1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol3.valuefunction, desired, 0.001);

}

BOOST_AUTO_TEST_CASE(test_randomized_mdp){
    RMDP rmdp;

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


    // *** ROBUST ******************
    // *** 2.0 ***
    test_randomized_threshold_robust(rmdp, 2.0, robust_2_0);

    // *** 1.0 ***
    test_randomized_threshold_robust(rmdp, 1.0, robust_1_0);

    // *** 0.5 ***
    test_randomized_threshold_robust(rmdp, 0.5, robust_0_5);

    // *** 0.0 ***
    test_randomized_threshold_robust(rmdp, 0.0, robust_0_0);

    // *** average ***
    // should be the same for the average
    test_randomized_threshold_average(rmdp, robust_0_0);


    // *** OPTIMISTIC ******************

    // *** 2.0 ***
    test_randomized_threshold_optimistic(rmdp, 2.0, optimistic_2_0);

    // *** 1.0 ***
    test_randomized_threshold_optimistic(rmdp, 1.0, optimistic_1_0);

    // *** 0.5 ***
    test_randomized_threshold_optimistic(rmdp, 0.5, optimistic_0_5);

    // *** 0.0 ***
    test_randomized_threshold_optimistic(rmdp, 0.0, optimistic_0_0);

}


// ********************************************************************************
// ***** Test terminal state ******************************************************
// ********************************************************************************

BOOST_AUTO_TEST_CASE(test_randomized_mdp_with_terminal_state){
    RMDP rmdp;

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


    // *** ROBUST ******************
    // *** 2.0 ***
    test_randomized_threshold_robust(rmdp, 2.0, robust_2_0);

    // *** 1.0 ***
    test_randomized_threshold_robust(rmdp, 1.0, robust_1_0);

    // *** 0.5 ***
    test_randomized_threshold_robust(rmdp, 0.5, robust_0_5);

    // *** 0.0 ***
    test_randomized_threshold_robust(rmdp, 0.0, robust_0_0);

    // *** average ***
    // should be the same for the average
    test_randomized_threshold_average(rmdp, robust_0_0);


    // *** OPTIMISTIC ******************

    // *** 2.0 ***
    test_randomized_threshold_optimistic(rmdp, 2.0, optimistic_2_0);

    // *** 1.0 ***
    test_randomized_threshold_optimistic(rmdp, 1.0, optimistic_1_0);

    // *** 0.5 ***
    test_randomized_threshold_optimistic(rmdp, 0.5, optimistic_0_5);

    // *** 0.0 ***
    test_randomized_threshold_optimistic(rmdp, 0.0, optimistic_0_0);

}

// ********************************************************************************
//          Test adding outcomes to a weighted action
// ********************************************************************************

BOOST_AUTO_TEST_CASE(test_create_outcome){

    WeightedOutcomeAction a;
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

    RMDP rmdp;

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
}


// ********************************************************************************
//  Test robustification 
// ********************************************************************************


MDP create_test_mdp_robustify(){
    MDP mdp(4);

    // nonrobust, single action, just to check basic robustification
    add_transition(mdp,0,0,1,0.5,1.0);
    add_transition(mdp,0,0,2,0.5,2.0);
    // probability of transition to state 3 is 0
    //add_transition<Model>(mdp,0,0,2,0.0,1.1);
    // states 1-4 are terminal (value 0)

    return mdp;
}

#if __cplusplus >= 201703L

BOOST_AUTO_TEST_CASE(test_robustification){
    MDP mdp = create_test_mdp_robustify();
    
    // no transition to zero probability states
    RMDP rmdp_nz = robustify(mdp, false);
    // allow transitions to zero probability states
    RMDP rmdp_z = robustify(mdp, true);

    // **** Test ordinary
    BOOST_CHECK_CLOSE(mpi_jac(mdp, 0.9).valuefunction[0],
                    (1.0 + 2.0) / 2.0, 1e-4);
    BOOST_CHECK_CLOSE(mpi_jac(rmdp_nz, 0.9).valuefunction[0],
                    (1.0 + 2.0) / 2.0, 1e-4);
    BOOST_CHECK_CLOSE(mpi_jac(rmdp_z, 0.9).valuefunction[0],
                    (1.0 + 2.0) / 2.0, 1e-4);
    
    // **** Test robust

    // robust MDP should have the same result as a robustified RMDP
    BOOST_CHECK_CLOSE(mpi_jac(mdp, 0.9, numvec(0), SARobustBellman(nats::robust_l1u(0.5))).valuefunction[0],
                    (1.0 * (0.5 + 0.25) + 2.0 * (0.5 - 0.25)), 1e-4);
    BOOST_CHECK_CLOSE(mpi_jac(rmdp_nz, 0.9, numvec(0), SARobustBellman<WeightedRobustState>(nats::robust_l1u(0.5))).valuefunction[0],
                    (1.0 * (0.5 + 0.25) + 2.0 * (0.5 - 0.25)), 1e-4);
    BOOST_CHECK_CLOSE(mpi_jac(rmdp_z, 0.9, numvec(0), SARobustBellman<WeightedRobustState>(nats::robust_l1u(0.5)) ).valuefunction[0],
                    (1.0 * (0.5) + 2.0 * (0.5 - 0.25) + 0.0 * 0.25), 1e-4);
}

#endif

// ********************************************************************************
//  Test s-rectangular MDP
// ********************************************************************************

BOOST_AUTO_TEST_CASE(s_rectangular){

    MDP mdp(3);

    add_transition(mdp,0,1,1,0.4,1.0);
    add_transition(mdp,0,1,2,0.3,2.0);
    add_transition(mdp,0,1,3,0.3,3.0);

    add_transition(mdp,0,0,1,0.2,3.0);
    add_transition(mdp,0,0,2,0.4,2.0);
    add_transition(mdp,0,0,3,0.4,1.0);

    add_transition(mdp,0,2,1,0.6,3.0);
    add_transition(mdp,0,2,2,0.4,2.0);

    add_transition(mdp,0,3,2,1.0,1.0);

    rsolve_vi(mdp, 1.0, nats::robust_s_l1(numvec{0.1,0,0,0}));
}

#ifdef GUROBI_USE

BOOST_AUTO_TEST_CASE(s_rectangular_gurobi){

    MDP mdp(3);

    add_transition(mdp,0,1,1,0.4,1.0);
    add_transition(mdp,0,1,2,0.3,2.0);
    add_transition(mdp,0,1,3,0.3,3.0);

    add_transition(mdp,0,0,1,0.2,3.0);
    add_transition(mdp,0,0,2,0.4,2.0);
    add_transition(mdp,0,0,3,0.4,1.0);

    add_transition(mdp,0,2,1,0.6,3.0);
    add_transition(mdp,0,2,2,0.4,2.0);

    add_transition(mdp,0,3,2,1.0,1.0);

    rsolve_vi(mdp, 1.0, nats::robust_s_l1_gurobi(numvec{0.1,0,0,0}));
}

#endif

// ********************************************************************************
//  Test optimization methods
// ********************************************************************************

#ifdef GUROBI_USE
GRBEnv& get_gurobi(){
    try{
        static GRBEnv env = GRBEnv();
        env.set(GRB_IntParam_OutputFlag, 0);
        return env;
    }catch(exception& e){
        cerr << "Problem constructing Gurobi object: " << endl
             << e.what() << endl;
        throw e;
    }catch(...){
        cerr << "Unknown exception while creating a gurobi object. Could be a license problem." << endl;
        throw;
    }

}
#endif

BOOST_AUTO_TEST_CASE(test_piecewise_linear_f){
    // make sure that the piecewise linear function is indeed linear between knots.

    // the nominal probability distribution
    const numvec p{0.3, 0.2, 0.1, 0.4};
    const numvec z{3.0, 2.0, 4.0, 1.0};

    numvec knots = worstcase_l1_knots(z,p).first;
    //std::cout << "knots = " << knots << std::endl;

    // make sure that each value between the knots is indeed a linear function
    for(size_t i = 0; i < knots.size() - 1; i++){
        double knot1 = knots[i];
        double knot2 = knots[i+1];

        double val1 = worstcase_l1_deviation(z,p,knot1).second;
        double val2 = worstcase_l1_deviation(z,p,knot2).second;

        for(double alpha : linspace(0,1,100)){
            double computed = worstcase_l1_deviation(z,p,alpha*knot1 + (1-alpha)*knot2).second;
            double expected = alpha*val1 + (1-alpha)*val2;

            BOOST_CHECK_CLOSE(computed, expected, 1e-5);
        }
    }
}

#if __cplusplus >= 201703L
#ifdef GUROBI_USE
BOOST_AUTO_TEST_CASE(test_solve_srect){
    // set parameters
    const vector<numvec> p{{0.3, 0.2, 0.1, 0.4},
                           {0.3, 0.6, 0.1},
                           {0.1, 0.3, 0.6}};
    const vector<numvec> z{{3.0, 2.0, 4.0, 1.0},
                           {3.0, 1.3, 4.0},
                           {6.0, 0.3, 4.5}};
    const vector<numvec> w{{0.3,0.3,0.3,0.1},
                           {0.2,0.5,0.3},
                           {0.7,0.1,0.2}};

    // uniform weights
    const vector<numvec> wu{{1.0,1.0,1.0,1.0},
                            {1.0,1.0,1.0},
                            {1.0,1.0,1.0}};


    GRBEnv env = get_gurobi();

    for(double psi = 0.0; psi < 3.0; psi += 0.1){
        auto [obj, d, xi] = solve_srect_bisection(z, p, psi, numvec(0), w);
        auto [gd, gobj] = srect_solve_gurobi(env, z, p, psi, w);

        // xi values can be smaller if actions are not active.
        BOOST_CHECK_GE(psi+1e-5, accumulate(xi.cbegin(), xi.cend(), 0.0));

        // compute static value
        // make sure that xi values are correct
        double expected_result = 0;
        for(size_t i = 0; i < z.size(); i++){
            numvec x = worstcase_l1_w(z[i], p[i], w[i], xi[i]).first;
            expected_result += d[i] * inner_product(x.cbegin(), x.cend(), z[i].cbegin(), 0.0);
        }

        BOOST_CHECK_CLOSE(obj, gobj, 1e-3);
        BOOST_CHECK_CLOSE(obj, expected_result, 1e-3);
        CHECK_CLOSE_COLLECTION(d, gd, 1e-3);
    }
}
#endif

BOOST_AUTO_TEST_CASE(test_responses){
    // set parameters
    const vector<numvec> p{{0.3, 0.2, 0.1, 0.4},
                           {0.3, 0.6, 0.1},
                           {0.1, 0.3, 0.6}};
    const vector<numvec> z{{3.0, 2.0, 4.0, 1.0},
                           {3.0, 1.3, 4.0},
                           {6.0, 0.3, 4.5}};
    const vector<numvec> w{{0.3,0.3,0.3,0.1},
                           {0.2,0.5,0.3},
                           {0.7,0.1,0.2}};

    // uniform weights
    const vector<numvec> wu{{1.0,1.0,1.0,1.0},
                            {1.0,1.0,1.0},
                            {1.0,1.0,1.0}};

    #ifdef GUROBI_USE
    GRBEnv env = get_gurobi();
    #endif

    for(size_t i = 0; i < p.size(); i++){
        const auto& pi = p[i];
        const auto& zi = z[i];
        const auto& wi = w[i];
        const auto& wui = wu[i];

        auto expected = inner_product(pi.cbegin(), pi.cend(), zi.cbegin(), 0.0);
        auto min = *min_element(zi.cbegin(), zi.cend());

        // psi = 0
        {
        double psi = 0;
        auto [pol,obj] = worstcase_l1(zi,pi,psi);
        auto [pol_w,obj_w] = worstcase_l1_w(zi,pi,wi,psi);
        auto [pol_wu,obj_wu] = worstcase_l1_w(zi,pi,wui,psi);
        CHECK_CLOSE_COLLECTION(pol, pol_w, 1e-3);
        CHECK_CLOSE_COLLECTION(pol, pol_wu, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_w, 1e-4);
        BOOST_CHECK_CLOSE(obj, obj_wu, 1e-4);
        BOOST_CHECK_CLOSE(obj, expected, 1e-4);
        }

        // psi = 1
        {
        double psi = 1;
        auto [pol,obj] = worstcase_l1(zi,pi,psi);
        auto [pol_w,obj_w] = worstcase_l1_w(zi,pi,wi,psi);
        auto [pol_wu,obj_wu] = worstcase_l1_w(zi,pi,wui,psi);
        CHECK_CLOSE_COLLECTION(pol, pol_wu, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_wu, 1e-4);

        #ifdef GUROBI_USE
        auto [pol_g,obj_g] = worstcase_l1_w_gurobi(env,zi,pi,wui,psi);
        auto [pol_w_g,obj_w_g] = worstcase_l1_w_gurobi(env,zi,pi,wi,psi);

        CHECK_CLOSE_COLLECTION(pol, pol_g, 1e-3);
        CHECK_CLOSE_COLLECTION(pol_w, pol_w_g, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_g, 1e-4);
        BOOST_CHECK_CLOSE(obj_w, obj_w, 1e-4);
        #endif
        }
        // psi = 2
        {
        double psi = 2;
        auto [pol,obj] = worstcase_l1(zi,pi,psi);
        auto [pol_w,obj_w] = worstcase_l1_w(zi,pi,wi,psi);
        auto [pol_wu,obj_wu] = worstcase_l1_w(zi,pi,wui,psi);
        CHECK_CLOSE_COLLECTION(pol, pol_wu, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_wu, 1e-4);
        BOOST_CHECK_CLOSE(obj, min, 1e-4);

        #ifdef GUROBI_USE
        auto [pol_g,obj_g] = worstcase_l1_w_gurobi(env,zi,pi,wui,psi);
        auto [pol_w_g,obj_w_g] = worstcase_l1_w_gurobi(env,zi,pi,wi,psi);

        CHECK_CLOSE_COLLECTION(pol, pol_g, 1e-3);
        CHECK_CLOSE_COLLECTION(pol_w, pol_w_g, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_g, 1e-4);
        BOOST_CHECK_CLOSE(obj_w, obj_w, 1e-4);
        #endif

        }
    }
}

BOOST_AUTO_TEST_CASE(test_responses_ties){
    // set parameters
    const vector<numvec> p{{0.3, 0.2, 0.1, 0.4},
                           {0.3, 0.6, 0.1},
                           {0.1, 0.3, 0.6}};
    const vector<numvec> z{{1.0, 1.0, 4.0, 1.0},
                           {1.0, 1.0, 1.0},
                           {1.0, 1.0, 1.0}};
    const vector<numvec> w{{0.3,0.3,0.3,0.1},
                           {0.2,0.5,0.3},
                           {0.7,0.1,0.2}};

    // uniform weights
    const vector<numvec> wu{{1.0,1.0,1.0,1.0},
                            {1.0,1.0,1.0},
                            {1.0,1.0,1.0}};

    #ifdef GUROBI_USE
    GRBEnv env = get_gurobi();
    #endif

    for(size_t i = 0; i < p.size(); i++){
        const auto& pi = p[i];
        const auto& zi = z[i];
        const auto& wi = w[i];
        const auto& wui = wu[i];

        auto expected = inner_product(pi.cbegin(), pi.cend(), zi.cbegin(), 0.0);
        auto min = *min_element(zi.cbegin(), zi.cend());

        // psi = 0
        {
        double psi = 0;
        auto [pol,obj] = worstcase_l1(zi,pi,psi);
        auto [pol_w,obj_w] = worstcase_l1_w(zi,pi,wi,psi);
        auto [pol_wu,obj_wu] = worstcase_l1_w(zi,pi,wui,psi);
        //CHECK_CLOSE_COLLECTION(pol, pol_w, 1e-3);
        //CHECK_CLOSE_COLLECTION(pol, pol_wu, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_w, 1e-4);
        BOOST_CHECK_CLOSE(obj, obj_wu, 1e-4);
        BOOST_CHECK_CLOSE(obj, expected, 1e-4);
        }

        // psi = 1
        {
        double psi = 1;
        auto [pol,obj] = worstcase_l1(zi,pi,psi);
        auto [pol_w,obj_w] = worstcase_l1_w(zi,pi,wi,psi);
        auto [pol_wu,obj_wu] = worstcase_l1_w(zi,pi,wui,psi);
        //CHECK_CLOSE_COLLECTION(pol, pol_wu, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_wu, 1e-4);

        #ifdef GUROBI_USE
        auto [pol_g,obj_g] = worstcase_l1_w_gurobi(env,zi,pi,wui,psi);
        auto [pol_w_g,obj_w_g] = worstcase_l1_w_gurobi(env,zi,pi,wi,psi);

        //CHECK_CLOSE_COLLECTION(pol, pol_g, 1e-3);
        //CHECK_CLOSE_COLLECTION(pol_w, pol_w_g, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_g, 1e-4);
        BOOST_CHECK_CLOSE(obj_w, obj_w, 1e-4);
        #endif
        }
        // psi = 2
        {
        double psi = 2;
        auto [pol,obj] = worstcase_l1(zi,pi,psi);
        auto [pol_w,obj_w] = worstcase_l1_w(zi,pi,wi,psi);
        auto [pol_wu,obj_wu] = worstcase_l1_w(zi,pi,wui,psi);
        //CHECK_CLOSE_COLLECTION(pol, pol_wu, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_wu, 1e-4);
        BOOST_CHECK_CLOSE(obj, min, 1e-4);

        #ifdef GUROBI_USE
        auto [pol_g,obj_g] = worstcase_l1_w_gurobi(env,zi,pi,wui,psi);
        auto [pol_w_g,obj_w_g] = worstcase_l1_w_gurobi(env,zi,pi,wi,psi);

        //CHECK_CLOSE_COLLECTION(pol, pol_g, 1e-3);
        //CHECK_CLOSE_COLLECTION(pol_w, pol_w_g, 1e-3);
        BOOST_CHECK_CLOSE(obj, obj_g, 1e-4);
        BOOST_CHECK_CLOSE(obj_w, obj_w, 1e-4);
        #endif

        }
    }
}


BOOST_AUTO_TEST_CASE(test_knots_deviation){
    // the nominal probability distribution
    const numvec p{0.3, 0.2, 0.1, 0.4};
    const numvec z{3.0, 2.0, 4.0, 1.0};

    numvec knots = worstcase_l1_knots(z,p).first;

    //cout << "knots = " << knots << endl;

    // make sure that each value between the knots is indeed a linear function
    for(size_t i = 0; i < knots.size() - 1; i++){
        double knot1 = knots[i];
        double knot2 = knots[i+1];

        double val1 = worstcase_l1_deviation(z,p,knot1).second;
        double val2 = worstcase_l1_deviation(z,p,knot2).second;

        for(double alpha = 0; alpha <= 0.99; alpha += 0.1){
            double val = worstcase_l1_deviation(z,p,alpha*knot1 + (1-alpha)*knot2).second;
               //cout << alpha << endl;
            BOOST_CHECK_CLOSE((alpha*val1 + (1-alpha)*val2), val, 1e-3);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_knots){
    // the nominal probability distribution
    const numvec p{0.3, 0.2, 0.1, 0.4};
    const numvec z{3.0, 2.0, 4.0, 1.0};

    numvec knots = worstcase_l1_knots(z,p).second;

    //cout << "knots = " << knots << endl;

    // make sure that each value between the knots is indeed a linear function
    for(size_t i = 0; i < knots.size() - 1; i++){
        double knot1 = knots[i];
        double knot2 = knots[i+1];

        double val1 = worstcase_l1(z,p,knot1).second;
        double val2 = worstcase_l1(z,p,knot2).second;

        for(double alpha = 0; alpha <= 1; alpha += 0.1){
            double val = worstcase_l1(z,p,alpha*knot1 + (1-alpha)*knot2).second;
               //cout << alpha << endl;
            BOOST_CHECK_CLOSE((alpha*val1 + (1-alpha)*val2), val, 1e-3);
        }
    }
}

BOOST_AUTO_TEST_CASE(test_knots_w){
    // the nominal probability distribution
    const numvec p{0.3, 0.2, 0.1, 0.4};
    const numvec z{3.0, 2.0, 4.0, 1.0};
    const numvec w{2.0, 2.0, 3.0, 1.0};

    numvec knots = worstcase_l1_w_knots(z,p,w).second;

    //cout << "knots = " << knots << endl;

    // make sure that each value between the knots is indeed a linear function
    for(size_t i = 0; i < knots.size() - 1; i++){
        double knot1 = knots[i];
        double knot2 = knots[i+1];

        double val1 = worstcase_l1_w(z,p,w,knot1).second;
        double val2 = worstcase_l1_w(z,p,w,knot2).second;

        for(double alpha = 0; alpha <= 1; alpha += 0.1){
            double val = worstcase_l1_w(z,p,w,alpha*knot1 + (1-alpha)*knot2).second;
               //cout << alpha << endl;
            BOOST_CHECK_CLOSE((alpha*val1 + (1-alpha)*val2), val, 1e-3);
        }
    }
}

/// tests weighted knots
BOOST_AUTO_TEST_CASE(test_knots_wu){
    // the nominal probability distribution
    const numvec p{0.3, 0.2, 0.1, 0.4};
    const numvec z{3.0, 2.0, 4.0, 1.0};
    const numvec w{1.0, 1.0, 1.0, 1.0};

    auto [knots,values] = worstcase_l1_knots(z,p);
    auto [knots_w,values_w] = worstcase_l1_w_knots(z,p,w);

    CHECK_CLOSE_COLLECTION(knots, knots_w, 1e-5);
    CHECK_CLOSE_COLLECTION(values, values_w, 1e-5);
}

#endif //__cplusplus >= 2017




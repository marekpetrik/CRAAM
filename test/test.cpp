#include <iostream>
#include <sstream>

#include "Transition.hpp"
#include "Action.hpp"
#include "State.hpp"
#include "RMDP.hpp"

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

BOOST_AUTO_TEST_CASE( basic_tests ) {
    Transition t1({1,2}, {0.1,0.2}, {3,4});
    Transition t2({1,2}, {0.1,0.2}, {5,4});
    Transition t3({1,2}, {0.1,0.3}, {3,4});

    // check value computation
    vector<prec_t> valuefunction = {0,1,2};
    auto ret = t1.compute_value(valuefunction,0.1);
    BOOST_CHECK (-0.01 <= ret - 1.15);
    BOOST_CHECK (ret - 1.15 <= 0.01);

    // check maximum selection
    Action a1({t1,t2}),a2({t1,t3});
    BOOST_CHECK(a1.maximal(valuefunction, 0.9).first == 1);
    BOOST_CHECK(a1.minimal(valuefunction,0.9).first == 0);
    BOOST_CHECK(a2.maximal(valuefunction, 0.9).first == 1);
    BOOST_CHECK(a2.minimal(valuefunction,0.9).first == 0);

    Action a3({t2});
    State s1({a1,a2,a3});
    auto v1 = get<2>(s1.max_max(valuefunction,0.9));
    auto v2 = get<2>(s1.max_min(valuefunction,0.9));
    BOOST_CHECK( -0.01 <= v1-2.13);
    BOOST_CHECK (v1-2.13 <= 0.01);
    BOOST_CHECK (-0.01 <= v2-1.75);
    BOOST_CHECK (v2-1.75 <= 0.01);
}

BOOST_AUTO_TEST_CASE(simple_mdp_vi_nonrobust) {
    RMDP rmdp(3);

    // nonrobust
    rmdp.add_transition_d(0,1,1,1,0);
    rmdp.add_transition_d(1,1,2,1,0);
    rmdp.add_transition_d(2,1,2,1,1.1);

    rmdp.add_transition_d(0,0,0,1,0);
    rmdp.add_transition_d(1,0,0,1,1);
    rmdp.add_transition_d(2,0,1,1,1);


    vector<prec_t> initial{0,0,0};

    // small number of iterations
    auto&& re = rmdp.vi_gs_rob(initial,0.9,20,0);

    vector<prec_t> val_rob{7.68072,8.67072,9.77072};
    vector<long> pol_rob{1,1,1};
    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_rob(initial,0.9, 20,0);

    vector<prec_t> val_rob2{7.5726,8.56265679,9.66265679};
    CHECK_CLOSE_COLLECTION(val_rob2,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // many iterations
    vector<prec_t> val_rob3{8.91,9.9,11.0};

    // robust
    re = rmdp.vi_gs_rob(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_rob(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());


    // optimistic
    re = rmdp.vi_gs_opt(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_opt(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // average
    re = rmdp.vi_gs_ave(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_ave(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_mpi_like_vi) {
    // run mpi but use parameters that should recover the same solution as vi

    RMDP rmdp(3);

    // nonrobust
    rmdp.add_transition_d(0,1,1,1,0);
    rmdp.add_transition_d(1,1,2,1,0);
    rmdp.add_transition_d(2,1,2,1,1.1);

    rmdp.add_transition_d(0,0,0,1,0);
    rmdp.add_transition_d(1,0,0,1,1);
    rmdp.add_transition_d(2,0,1,1,1);


    vector<prec_t> initial{0,0,0};

    // small number of iterations
    auto&& re = rmdp.vi_gs_rob(initial,0.9,20,0);

    vector<prec_t> val_rob{7.68072,8.67072,9.77072};
    vector<long> pol_rob{1,1,1};
    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.mpi_jac_rob(initial,0.9, 20,0, 0,0);

    vector<prec_t> val_rob2{7.5726,8.56265679,9.66265679};
    CHECK_CLOSE_COLLECTION(val_rob2,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // many iterations
    vector<prec_t> val_rob3{8.91,9.9,11.0};

    // robust
    re = rmdp.vi_gs_rob(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_rob(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());


    // optimistic
    re = rmdp.vi_gs_opt(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_opt(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // average
    re = rmdp.vi_gs_ave(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    re = rmdp.vi_jac_ave(initial,0.9, 10000,0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_mpi_nonrobust) {
    RMDP rmdp(3);

    // nonrobust
    rmdp.add_transition_d(0,1,1,1,0);
    rmdp.add_transition_d(1,1,2,1,0);
    rmdp.add_transition_d(2,1,2,1,1.1);

    rmdp.add_transition_d(0,0,0,1,0);
    rmdp.add_transition_d(1,0,0,1,1);
    rmdp.add_transition_d(2,0,1,1,1);


    vector<prec_t> initial{0,0,0};

    // many iterations
    vector<prec_t> val_rob3{8.91,9.9,11.0};
    vector<long> pol_rob{1,1,1};

    // robust
    auto re = rmdp.mpi_jac_rob(initial,0.9, 100,0, 100, 0);
    CHECK_CLOSE_COLLECTION(val_rob3,re.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

}

BOOST_AUTO_TEST_CASE(simple_mdp_fixed_size) {
    RMDP rmdp(2);

    rmdp.add_transition_d(0,1,1,1,0);

    rmdp.add_transition_d(1,1,2,1,0);

}

BOOST_AUTO_TEST_CASE(check_add_transition) {

    RMDP rmdp(10);

    // check adding to the end
    rmdp.add_transition(0,0,0,5,0.1,1);
    rmdp.add_transition(0,0,0,7,0.1,2);

    const Transition& transition = rmdp.get_transition(0,0,0);

    BOOST_CHECK(is_sorted(transition.indices.begin(), transition.indices.end()) );
    BOOST_CHECK_EQUAL(transition.indices.size(), 2);
    BOOST_CHECK_EQUAL(transition.probabilities.size(), 2);
    BOOST_CHECK_EQUAL(transition.rewards.size(), 2);

    // check updating the last element
    rmdp.add_transition(0,0,0,7,0.4,4);

    BOOST_CHECK(is_sorted(transition.indices.begin(), transition.indices.end()) );
    BOOST_CHECK_EQUAL(transition.indices.size(), 2);
    BOOST_CHECK_EQUAL(transition.probabilities.size(), 2);
    BOOST_CHECK_EQUAL(transition.rewards.size(), 2);
    vector<double> tr{1.0,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.rewards.begin(), transition.rewards.end(), tr.begin(), tr.end());
    vector<double> tp{0.1,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.probabilities.begin(), transition.probabilities.end(), tp.begin(), tp.end());


    // check inserting an element into the middle
    rmdp.add_transition(0,0,0,6,0.1,0.5);

    BOOST_CHECK(is_sorted(transition.indices.begin(), transition.indices.end()) );
    BOOST_CHECK_EQUAL(transition.indices.size(), 3);
    BOOST_CHECK_EQUAL(transition.probabilities.size(), 3);
    BOOST_CHECK_EQUAL(transition.rewards.size(), 3);
    tr = vector<double>{1.0,0.5,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.rewards.begin(), transition.rewards.end(), tr.begin(), tr.end());
    tp = vector<double>{0.1,0.1,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.probabilities.begin(), transition.probabilities.end(), tp.begin(), tp.end());


    // check updating an element in the middle
    rmdp.add_transition(0,0,0,6,0.1,1.5);

    BOOST_CHECK(is_sorted(transition.indices.begin(), transition.indices.end()) );
    BOOST_CHECK_EQUAL(transition.indices.size(), 3);
    BOOST_CHECK_EQUAL(transition.probabilities.size(), 3);
    BOOST_CHECK_EQUAL(transition.rewards.size(), 3);
    tr = vector<double>{1.0,1.0,3.6};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.rewards.begin(), transition.rewards.end(), tr.begin(), tr.end());
    tp = vector<double>{0.1,0.2,0.5};
    BOOST_CHECK_EQUAL_COLLECTIONS(transition.probabilities.begin(), transition.probabilities.end(), tp.begin(), tp.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_resize) {
    RMDP rmdp(0);

    rmdp.add_transition_d(0,1,1,1,0);
    rmdp.add_transition_d(1,1,2,1,0);
    rmdp.add_transition_d(2,1,2,1,1.1);

    rmdp.add_transition_d(0,0,0,1,0);
    rmdp.add_transition_d(1,0,0,1,1);
    rmdp.add_transition_d(2,0,1,1,1);

    vector<prec_t> initial(3);
    for(auto & i : initial)
        i = 0;

    auto&& re = rmdp.vi_gs_rob(initial,0.9, 20l,0);

    vector<prec_t> val_rob{7.68072,8.67072,9.77072};
    vector<long> pol_rob{1,1,1};
    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());
}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load) {
    RMDP rmdp1(0);

    rmdp1.add_transition_d(0,1,1,1,0);
    rmdp1.add_transition_d(1,1,2,1,0);
    rmdp1.add_transition_d(2,1,2,1,1.1);

    rmdp1.add_transition_d(0,0,0,1,0);
    rmdp1.add_transition_d(1,0,0,1,1);
    rmdp1.add_transition_d(2,0,1,1,1);


    stringstream store;

    rmdp1.transitions_to_csv(store);
    store.seekg(0);

    auto rmdp = RMDP::transitions_from_csv(store);

    vector<prec_t> initial(3);
    for(auto & i : initial)
        i = 0;

    auto&& re = rmdp->vi_gs_rob(initial,0.9, 20l,0);

    vector<prec_t> val_rob{7.68072,8.67072,9.77072};
    vector<long> pol_rob{1,1,1};
    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

}

BOOST_AUTO_TEST_CASE(simple_mdp_save_load_save_load) {
    RMDP rmdp1(0);

    rmdp1.add_transition_d(0,1,1,1,0);
    rmdp1.add_transition_d(1,1,2,1,0);
    rmdp1.add_transition_d(2,1,2,1,1.1);

    rmdp1.add_transition_d(0,0,0,1,0);
    rmdp1.add_transition_d(1,0,0,1,1);
    rmdp1.add_transition_d(2,0,1,1,1);

    stringstream store;

    rmdp1.transitions_to_csv(store);
    store.seekg(0);

    auto&& string1 = store.str();
    auto rmdp2 = RMDP::transitions_from_csv(store);

    stringstream store2;

    rmdp2->transitions_to_csv(store2);

    auto&& string2 = store2.str();

    BOOST_CHECK_EQUAL(string1, string2);
}

BOOST_AUTO_TEST_CASE( test_value_function ) {
    RMDP rmdp(1);

    rmdp.add_transition(0,0,0,0,1,1);
    rmdp.add_transition(0,0,1,0,1,2);

    vector<prec_t> initial(1);
    for(auto & i : initial)
        i = 0;

    auto&& result1 = rmdp.vi_gs_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = rmdp.vi_gs_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);
}

BOOST_AUTO_TEST_CASE(test_value_function_robust_optimistic){
    RMDP rmdp(1);

    rmdp.add_transition(0,0,0,0,1,1);
    rmdp.add_transition(0,0,1,0,1,2);

    vector<prec_t> initial(1);
    for(auto & i : initial)
        i = 0;

    auto&& result1 = rmdp.vi_jac_rob(initial,0.9, 1000000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = rmdp.vi_jac_opt(initial,0.9, 1000000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0],20,1e-3);

    auto&& result3 = rmdp.vi_jac_ave(initial,0.9, 1000000, 0);
    BOOST_CHECK_CLOSE(result3.valuefunction[0],15,1e-3);
}

BOOST_AUTO_TEST_CASE(test_l1_worst_case){
    vector<prec_t> q = {0.4, 0.3, 0.1, 0.2};
    vector<prec_t> z = {1.0, 2.0, 5.0, 4.0};
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

    vector<prec_t> q1 = {1.0};
    vector<prec_t> z1 = {2.0};

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
};

BOOST_AUTO_TEST_CASE(test_value_function_l1){
    RMDP rmdp(1);

    vector<prec_t> dist = {0.5,0.5};

    rmdp.add_transition(0,0,0,0,1,1);
    rmdp.add_transition(0,0,1,0,1,2);
    rmdp.set_distribution(0,0,dist,2);

    vector<prec_t> initial = {0};

    auto&& result1 = rmdp.vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = rmdp.vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    rmdp.set_uniform_thresholds(0);
    result1 = rmdp.vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0],15.0,1e-3);

    result2 = rmdp.vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0],15.0,1e-3);

    rmdp.set_uniform_thresholds(1);
    result1 = rmdp.vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0],10.0,1e-3);

    result2 = rmdp.vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    rmdp.set_uniform_thresholds(0.5);
    result1 = rmdp.vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0],12.5,1e-3);

    result2 = rmdp.vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0],17.5,1e-3);
}

BOOST_AUTO_TEST_CASE(test_string){
    RMDP rmdp(2);

    vector<prec_t> dist = {0.5,0.5};

    rmdp.add_transition(0,0,0,0,1,1);
    rmdp.add_transition(0,0,1,0,1,2);
    rmdp.set_distribution(0,0,dist,2);

    rmdp.add_transition(1,0,0,0,1,1);
    rmdp.add_transition(1,0,1,0,1,2);
    rmdp.set_distribution(1,0,dist,2);

    auto s = rmdp.to_string();
    BOOST_CHECK_EQUAL(s.length(), 40);
}

BOOST_AUTO_TEST_CASE(test_normalization) {
    RMDP rmdp(2);

    // nonrobust
    rmdp.add_transition_d(0,0,0,1.0,0.1);
    rmdp.add_transition_d(0,0,1,1.0,0.5);


    // make sure that the MDP is not reported to be normalized
    BOOST_CHECK(!rmdp.is_normalized());

    // make sure that the normalization forks
    rmdp.normalize();
    BOOST_CHECK(rmdp.is_normalized());

    // solve and check value function
    vector<prec_t> initial{0,0};
    auto&& re = rmdp.vi_jac_rob(initial,0.9,2000,0);

    vector<prec_t> val{0.545454545455, 0.0};

    CHECK_CLOSE_COLLECTION(val,re.valuefunction,1e-3);
}

BOOST_AUTO_TEST_CASE(test_randomized_mdp){
    RMDP m(5);

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
    vector<prec_t> robust_2_0 {0.9*20,20,30,10,40};
    vector<prec_t> robust_1_0 {0.9*20,20,30,10,40};
    vector<prec_t> robust_0_5 {0.9*90.0/4.0,20,30,10,40};
    vector<prec_t> robust_0_0 {0.9*25.0,20,30,10,40};
    vector<prec_t> optimistic_2_0 {0.9*40,20,30,10,40};
    vector<prec_t> optimistic_1_0 {0.9*40,20,30,10,40};
    vector<prec_t> optimistic_0_5 {0.9*130.0/4.0,20,30,10,40};
    vector<prec_t> optimistic_0_0 {0.9*25.0,20,30,10,40};

    stringstream store(string_representation);

    store.seekg(0);
    auto rmdp = RMDP::transitions_from_csv(store,false);

    // print the problem definition for debugging
    //cout << string_representation << endl;
    //cout << rmdp->state_count() << endl;
    //stringstream store2;
    //rmdp->transitions_to_csv(store2);
    //cout << store2.str() << endl;

    vector<prec_t> value(5,0.0);
    const prec_t gamma = 0.9;
    Solution sol;

    // *** ROBUST ******************

    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);

    // should be the same without the l1 bound
    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->vi_gs_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->mpi_jac_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);

    rmdp->set_uniform_distribution(1.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);

    rmdp->set_uniform_distribution(0.5);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);

    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);

    // should be the same for the average
    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_ave(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->vi_gs_ave(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->mpi_jac_ave(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);


    // *** OPTIMISTIC ******************

    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);

    // should be the same without the l1 bound
    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->vi_gs_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->mpi_jac_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);

    rmdp->set_uniform_distribution(1.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);

    rmdp->set_uniform_distribution(0.5);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);

    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
}

BOOST_AUTO_TEST_CASE(test_randomized_mdp_with_terminal_state){
    RMDP m(6);

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
    vector<prec_t> robust_2_0 {0.9*20,20,30,10,40,0};
    vector<prec_t> robust_1_0 {0.9*20,20,30,10,40,0};
    vector<prec_t> robust_0_5 {0.9*90.0/4.0,20,30,10,40,0};
    vector<prec_t> robust_0_0 {0.9*25.0,20,30,10,40,0};
    vector<prec_t> optimistic_2_0 {0.9*40,20,30,10,40,0};
    vector<prec_t> optimistic_1_0 {0.9*40,20,30,10,40,0};
    vector<prec_t> optimistic_0_5 {0.9*130.0/4.0,20,30,10,40,0};
    vector<prec_t> optimistic_0_0 {0.9*25.0,20,30,10,40,0};

    stringstream store(string_representation);

    store.seekg(0);
    auto rmdp = RMDP::transitions_from_csv(store,false);

    // print the problem definition for debugging
    //cout << string_representation << endl;
    //cout << rmdp->state_count() << endl;
    //stringstream store2;
    //rmdp->transitions_to_csv(store2);
    //cout << store2.str() << endl;

    vector<prec_t> value(6,0.0);
    const prec_t gamma = 0.9;
    Solution sol;

    // *** ROBUST ******************

    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);

    // should be the same without the l1 bound
    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->vi_gs_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->mpi_jac_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);

    rmdp->set_uniform_distribution(1.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);

    rmdp->set_uniform_distribution(0.5);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);

    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);

    // should be the same for the average
    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_ave(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->vi_gs_ave(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->mpi_jac_ave(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);


    // *** OPTIMISTIC ******************

    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);

    // should be the same without the l1 bound
    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->vi_gs_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->mpi_jac_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);

    rmdp->set_uniform_distribution(1.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);

    rmdp->set_uniform_distribution(0.5);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);

    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);

}

BOOST_AUTO_TEST_CASE(test_parameter_read_write){
    RMDP m(6);

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
    auto rmdp = RMDP::transitions_from_csv(store,false);

    BOOST_CHECK_EQUAL(rmdp->get_reward(3,0,0,0), 10.0);
    rmdp->set_reward(3,0,0,0,15.1);
    BOOST_CHECK_EQUAL(rmdp->get_reward(3,0,0,0), 15.1);

    BOOST_CHECK_EQUAL(rmdp->get_reward(0,1,0,1), 2.0);
    rmdp->set_reward(0,1,0,1,19.1);
    BOOST_CHECK_EQUAL(rmdp->get_reward(0,1,0,1), 19.1);

    BOOST_CHECK_EQUAL(rmdp->get_threshold(3,0), 0);
    rmdp->set_threshold(3,0,1.0);
    BOOST_CHECK_EQUAL(rmdp->get_threshold(3,0), 1.0);

}

BOOST_AUTO_TEST_CASE(test_rmdp_copy){
    RMDP rmdp_original(1);

    vector<prec_t> dist = {0.5,0.5};

    rmdp_original.add_transition(0,0,0,0,1,1);
    rmdp_original.add_transition(0,0,1,0,1,2);
    rmdp_original.set_distribution(0,0,dist,2);

    auto rmdp = rmdp_original.copy();

    vector<prec_t> initial = {0};

    auto&& result1 = rmdp->vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0], 10.0, 1e-3);

    auto&& result2 = rmdp->vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    rmdp->set_uniform_thresholds(0);
    result1 = rmdp->vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0],15.0,1e-3);

    result2 = rmdp->vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0],15.0,1e-3);

    rmdp->set_uniform_thresholds(1);
    result1 = rmdp->vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0],10.0,1e-3);

    result2 = rmdp->vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0], 20.0, 1e-3);

    rmdp->set_uniform_thresholds(0.5);
    result1 = rmdp->vi_gs_l1_rob(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result1.valuefunction[0],12.5,1e-3);

    result2 = rmdp->vi_gs_l1_opt(initial,0.9, 1000, 0);
    BOOST_CHECK_CLOSE(result2.valuefunction[0],17.5,1e-3);
}


BOOST_AUTO_TEST_CASE(test_mdp_copy2){
    RMDP m(6);

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
    vector<prec_t> robust_2_0 {0.9*20,20,30,10,40,0};
    vector<prec_t> robust_1_0 {0.9*20,20,30,10,40,0};
    vector<prec_t> robust_0_5 {0.9*90.0/4.0,20,30,10,40,0};
    vector<prec_t> robust_0_0 {0.9*25.0,20,30,10,40,0};
    vector<prec_t> optimistic_2_0 {0.9*40,20,30,10,40,0};
    vector<prec_t> optimistic_1_0 {0.9*40,20,30,10,40,0};
    vector<prec_t> optimistic_0_5 {0.9*130.0/4.0,20,30,10,40,0};
    vector<prec_t> optimistic_0_0 {0.9*25.0,20,30,10,40,0};

    stringstream store(string_representation);

    store.seekg(0);
    auto rmdp1 = RMDP::transitions_from_csv(store,false);

    // copying to make sure that copy works
    auto rmdp = rmdp1->copy();

    // change rewards to make sure that it is really a copy
    rmdp1->set_reward(1,0,0,0,9.0);

    // print the problem definition for debugging
    //cout << string_representation << endl;
    //cout << rmdp->state_count() << endl;
    //stringstream store2;
    //rmdp->transitions_to_csv(store2);
    //cout << store2.str() << endl;

    vector<prec_t> value(6,0.0);
    const prec_t gamma = 0.9;
    Solution sol;

    // *** ROBUST ******************

    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);

    // should be the same without the l1 bound
    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->vi_gs_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);
    sol = rmdp->mpi_jac_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_2_0, 0.001);

    rmdp->set_uniform_distribution(1.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_1_0, 0.001);

    rmdp->set_uniform_distribution(0.5);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_5, 0.001);

    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->vi_gs_l1_rob(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->mpi_jac_l1_rob(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);

    // should be the same for the average
    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_ave(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->vi_gs_ave(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);
    sol = rmdp->mpi_jac_ave(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, robust_0_0, 0.001);


    // *** OPTIMISTIC ******************

    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);

    // should be the same without the l1 bound
    rmdp->set_uniform_distribution(2.0);
    sol = rmdp->vi_jac_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->vi_gs_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);
    sol = rmdp->mpi_jac_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_2_0, 0.001);

    rmdp->set_uniform_distribution(1.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_1_0, 0.001);

    rmdp->set_uniform_distribution(0.5);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_5, 0.001);

    rmdp->set_uniform_distribution(0.0);
    sol = rmdp->vi_jac_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
    sol = rmdp->vi_gs_l1_opt(value,gamma,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);
    sol = rmdp->mpi_jac_l1_opt(value,gamma,1000,1e-5,1000,1e-5);
    CHECK_CLOSE_COLLECTION(sol.valuefunction, optimistic_0_0, 0.001);

}

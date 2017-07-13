#include "../craam/RMDP.hpp"
#include "../craam/modeltools.hpp"
#include "../craam/algorithms/valueiteration.hpp"
#include "../craam/algorithms/occupancies.hpp"

#include <iostream>
#include <sstream>
#include <cmath>
#include <numeric>
#include <utility>


using namespace std;
using namespace craam;
using namespace craam::algorithms;

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
    mpi_jac(m, 0.9);
    vi_gs(m, 0.9);
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
    BOOST_CHECK_CLOSE(value_action(a1, valuefunction, 0.9, make_pair(robust_unbounded<prec_t>,0.0)).second,
                                min(t1.value(valuefunction, 0.9), t2.value(valuefunction, 0.9)), 1e-3);
    BOOST_CHECK_CLOSE(value_action(a1, valuefunction, 0.9, make_pair(optimistic_unbounded<prec_t>,0.0)).second,
                                max(t1.value(valuefunction, 0.9), t2.value(valuefunction, 0.9)), 1e-3);

    WeightedRobustState s1({a1,a2,a3});
    auto v1 = get<2>(value_max_state(s1,valuefunction,0.9,make_pair(optimistic_unbounded<prec_t>, 0.0)));
    auto v2 = get<2>(value_max_state(s1,valuefunction,0.9,make_pair(robust_unbounded<prec_t>, 0.0)));
    BOOST_CHECK_CLOSE (v1, 2.13, 1e-3);
    BOOST_CHECK_CLOSE (v2, 1.75, 1e-3);
}

// ********************************************************************************
// ***** MDP value iteration ****************************************************
// ********************************************************************************

template<class Model>
void test_simple_vi(){
    // Tests simple non-robust value iteration with the various models
    auto rmdp = create_test_mdp<Model>();

    indvec natpol_rob{0,0,0};

    Transition init_d({0,1,2},{1.0/3.0,1.0/3.0,1.0/3.0},{0,0,0});

    numvec initial{0,0,0};

    indvec pol_rob{1,1,1};

    // small number of iterations (not the true value function)
    numvec val_rob{7.68072,8.67072,9.77072};
    auto re = vi_gs(rmdp,0.9,initial,PolicyDeterministic(indvec(0)),20,0);

    CHECK_CLOSE_COLLECTION(val_rob,re.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re.policy.begin(),re.policy.end());

    // test jac value iteration with small number of iterations ( not the true value function)
    auto re2 = mpi_jac(rmdp, 0.9, initial, PolicyDeterministic(indvec(0)), 20,0,0);

    numvec val_rob2{7.5726,8.56265679,9.66265679};
    CHECK_CLOSE_COLLECTION(val_rob2,re2.valuefunction,1e-3);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re2.policy.begin(),re2.policy.end());

    // many iterations
    const numvec val_rob3{8.91,9.9,11.0};
    const numvec occ_freq3{0.333333333,0.6333333333,9.03333333333333};
    const prec_t ret_true = inner_product(val_rob3.cbegin(), val_rob3.cend(), init_d.get_probabilities().cbegin(),0.0);

    // robust
    auto&& re3 = vi_gs(rmdp,0.9,robust_l1,0,initial);
    CHECK_CLOSE_COLLECTION(val_rob3,re3.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re3.policy.begin(),re3.policy.end());

    auto&& re4 = mpi_jac(rmdp,0.9,robust_l1, 0, initial, indvec(0), vector<numvec>(0), 1000, 0.0, 1000, 0.0, true);
    CHECK_CLOSE_COLLECTION(val_rob3,re4.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re4.policy.begin(),re4.policy.end());

    // optimistic
    auto&& re5 = vi_gs(rmdp,0.9, optimistic_l1, 0, initial);
    CHECK_CLOSE_COLLECTION(val_rob3,re5.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re5.policy.begin(),re5.policy.end());

    auto&& re6 = mpi_jac(rmdp,0.9, optimistic_l1, 0, initial);
    CHECK_CLOSE_COLLECTION(val_rob3,re6.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re6.policy.begin(),re6.policy.end());

    // plain
    auto&& re7 = vi_gs(rmdp, 0.9, initial);
    CHECK_CLOSE_COLLECTION(val_rob3,re7.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re7.policy.begin(),re7.policy.end());

    auto&& re8 = mpi_jac(rmdp, 0.9, initial);
    CHECK_CLOSE_COLLECTION(val_rob3,re8.valuefunction,1e-2);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol_rob.begin(),pol_rob.end(),re8.policy.begin(),re8.policy.end());

    // fixed evaluation
    auto&& re9 = mpi_jac(rmdp,0.9,initial,pol_rob, 10000,0.0, 0);
    CHECK_CLOSE_COLLECTION(val_rob3,re9.valuefunction,1e-2);

    // check the computed returns
    BOOST_CHECK_CLOSE (re8.total_return(init_d), ret_true, 1e-2);

    // check if we get the same return from the solution as from the
    // occupancy frequencies
    auto&& occupancy_freq = occfreq_mat(rmdp, init_d,0.9,re.policy);
    CHECK_CLOSE_COLLECTION(occupancy_freq, occ_freq3, 1e-3);

    auto&& rewards = rewards_vec(rmdp, re3.policy);
    auto cmp_tr = inner_product(rewards.begin(), rewards.end(), occupancy_freq.begin(), 0.0);
    BOOST_CHECK_CLOSE (cmp_tr, ret_true, 1e-3);
}

BOOST_AUTO_TEST_CASE(simple_mdp_vi_of_nonrobust) {
    test_simple_vi<MDP>();
}

//BOOST_AUTO_TEST_CASE(simple_rmdpd_vi_of_nonrobust) {
//    test_simple_vi<RMDP>();
//}

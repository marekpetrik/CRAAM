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

#include "craam/ImMDP.hpp"
#include "craam/Simulation.hpp"
#include "craam/Samples.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"

#include <rm/range.hpp>

#include <boost/functional/hash.hpp>
#include <iostream>
#include <iterator>
#include <cmath>

using namespace std;
using namespace craam;
using namespace craam::algorithms;
using namespace craam::impl;
using namespace util::lang;


/**
Creates a simple chain problem.
Actions:   0 - left
           1 - right
Optimal solution: Action 1, with value function:
    [1.1 gamma^2/(1-gamma), 1.1 gamma/(1-gamma), 1.1/(1-gamma)]

*/
MDP make_chain1(){
    MDP rmdp(3);

    add_transition(rmdp,0,1,1,1,0);
    add_transition(rmdp,1,1,2,1,0);
    add_transition(rmdp,2,1,2,1,1.1);

    add_transition(rmdp,0,0,0,1,0);
    add_transition(rmdp,1,0,0,1,1);
    add_transition(rmdp,2,0,1,1,1);

    return rmdp;
}

BOOST_AUTO_TEST_CASE( simple_construct_mdpi ) {

    auto mdp = make_shared<MDP>();
    vector<long> observations({0,0});
    Transition initial(vector<long>{0,1},vector<prec_t>{0.5,0.5},vector<prec_t>{0,0});

    add_transition(*mdp,0,0,1,1.0,1.0);
    add_transition(*mdp,1,0,0,1.0,1.0);
    BOOST_CHECK_EQUAL(mdp->state_count(), 2);

    MDPI im(const_pointer_cast<const MDP>(mdp), observations, initial);

    MDPI im2(*mdp,observations,initial);

    // check that we really have a copy
    add_transition(*mdp,1,0,2,1.0,1.0);

    BOOST_CHECK_EQUAL(mdp->state_count(), 3);
    BOOST_CHECK_EQUAL(im.get_mdp()->state_count(), 3);
    BOOST_CHECK_EQUAL(im2.get_mdp()->state_count(), 2);
}

BOOST_AUTO_TEST_CASE( simple_construct_mdpi_r ) {

    auto mdp = make_shared<MDP>();
    vector<long> observations({0,0});
    Transition initial(vector<long>{0,1},vector<prec_t>{0.5,0.5},vector<prec_t>{0,0});

    add_transition(*mdp,0,0,1,1.0,1.0);
    add_transition(*mdp,1,0,0,1.0,2.0);

    MDPI_R imr(const_pointer_cast<const MDP>(mdp), observations, initial);

    // COPY ! so we can change the threshold
    auto rmdp = imr.get_robust_mdp();

    BOOST_CHECK_EQUAL(rmdp.state_count(), 1);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).action_count(), 1);
    BOOST_CHECK_EQUAL(rmdp.get_state(0).get_action(0).outcome_count(), 2);

    vector<prec_t> iv(rmdp.state_count(),0.0);

    auto&& so = mpi_jac(rmdp, 0.9, iv, SARobustBellman<WeightedRobustState>(nats::optimistic_unbounded()), 100, 0.0, 10, 0.0);
    BOOST_CHECK_CLOSE(so.valuefunction[0], 20, 1e-3);

    auto&& sr = mpi_jac(rmdp,0.9,iv,SARobustBellman<WeightedRobustState>(nats::robust_unbounded()), 100,0.0,10,0.0);
    BOOST_CHECK_CLOSE(sr.valuefunction[0], 10, 1e-3);
}

BOOST_AUTO_TEST_CASE( small_construct_mdpi_r ) {

    auto mdp = make_shared<MDP>();
    vector<long> observations{0,0,1};

    Transition initial(vector<long>{0,1,2},vector<prec_t>{1.0/3.0,1.0/3.0,1.0/3.0},
                        vector<prec_t>{0,0,0});

    // action 0
    add_transition(*mdp,0,0,0,0.5,1.0);
    add_transition(*mdp,0,0,1,0.5,1.0);

    add_transition(*mdp,1,0,0,0.5,2.0);
    add_transition(*mdp,1,0,1,0.5,2.0);

    add_transition(*mdp,2,0,2,1.0,1.2);

    // action 1
    add_transition(*mdp,0,1,2,1.0,1.2);
    add_transition(*mdp,1,1,2,1.0,1.2);

    BOOST_TEST_CHECKPOINT("Constructing MDPI_R.");
    MDPI_R imr(const_pointer_cast<const MDP>(mdp), observations, initial);

    // Copy to change threshold
    auto rmdp = imr.get_robust_mdp();

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
    auto&& so = mpi_jac(rmdp,0.9,iv,SARobustBellman<WeightedRobustState>(nats::optimistic_unbounded()),100,0.0,10,0.0);
    CHECK_CLOSE_COLLECTION(so.valuefunction, target_v_opt, 1e-3);

    auto&& sr = mpi_jac(rmdp,0.9,iv,SARobustBellman<WeightedRobustState>(nats::robust_unbounded()),100,0.0,10,0.0);
    CHECK_CLOSE_COLLECTION(sr.valuefunction, target_v_rob, 1e-3);
}

BOOST_AUTO_TEST_CASE( small_reweighted_solution ) {

    auto mdp = make_shared<MDP>();
    vector<long> observations({0,0,1});
    Transition initial(vector<long>{0,1,2},vector<prec_t>{1.0/3.0,1.0/3.0,1.0/3.0},
                        vector<prec_t>{0,0,0});

    // action 0
    add_transition(*mdp,0,0,0,0.5,1.0);
    add_transition(*mdp,0,0,1,0.5,1.0);

    add_transition(*mdp,1,0,0,0.5,2.0);
    add_transition(*mdp,1,0,1,0.5,2.0);

    add_transition(*mdp,2,0,2,1.0,1.2);

    // action 1
    add_transition(*mdp,0,1,2,1.0,1.2);
    add_transition(*mdp,1,1,2,1.0,1.2);

    BOOST_TEST_CHECKPOINT("Constructing MDPI_R.");
    MDPI_R imr(const_pointer_cast<const MDP>(mdp), observations, initial);

    BOOST_TEST_CHECKPOINT("Solving MDPI_R.");
    auto&& pol = imr.solve_reweighted(10, 0.9);

    indvec polvec{0,0};
    BOOST_CHECK_EQUAL_COLLECTIONS(pol.begin(), pol.end(),polvec.begin(),polvec.end());

    auto&& pol2 = imr.solve_robust(10, 0.0, 0.9);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol2.begin(), pol2.end(),polvec.begin(),polvec.end());


    //auto retval = imr.total_return(pol, 0.99);
    //cout << "Return: " << retval << endl;

    //ostream_iterator<prec_t> output(cout, ", ");
    //copy(pol.begin(), pol.end(), output);
}

BOOST_AUTO_TEST_CASE(simple_mdpo_save_load_save_load) {
    MDP&& rmdp1 = make_chain1();

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
    MDP&& rmdp1 = make_chain1();

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

using namespace craam::msen;

/*
template<class T>
void print_vector(vector<T> vec){
    for(auto&& p : vec){
        cout << p << " ";
    }
}*/

#if __cplusplus >= 201703L

BOOST_AUTO_TEST_CASE(implementable_from_samples){
    const int terminal_state = 10;

    CounterTerminal sim(0.9,0,terminal_state,1);
    RandomPolicy<CounterTerminal> random_pol(sim,1);

    auto samples = make_samples<CounterTerminal>();
    simulate(sim,samples,random_pol,50,50);
    simulate(sim,samples,[](int){return 1;},10,20);
    simulate(sim,samples,[](int){return -1;},10,20);

    SampleDiscretizerSI<typename CounterTerminal::State, 
                        typename CounterTerminal::Action> sd;
    // initialize action values
    sd.add_action(-1); sd.add_action(+1);
    //initialize state values
    for(auto i : range(-terminal_state,terminal_state)) sd.add_state(i);

    sd.add_samples(samples);

    BOOST_CHECK_EQUAL(samples.get_initial().size(), sd.get_discrete()->get_initial().size());
    BOOST_CHECK_EQUAL(samples.size(), sd.get_discrete()->size());

    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());
    auto mdp = smdp.get_mdp();
    auto&& initial = smdp.get_initial();

    auto&& sol = mpi_jac(*mdp,0.9);

    //cout << "Optimal policy: " << endl; print_vector(sol.policy); cout << endl;

    BOOST_CHECK_CLOSE(sol.total_return(initial), 51.313973553, 1e-3);

    // define observations
    indvec observations(mdp->state_count(), -1);
    size_t last_obs(0), inobs(0);
    for(auto i : range(size_t(0), mdp->state_count())){
        // check if this is a terminal state
        if(mdp->get_state(i).action_count() == 0 || inobs >= 2){
            if(inobs > 0){
                inobs = 0;
                last_obs++;
            }
            observations[i] = last_obs++;
        }else {
            observations[i] = last_obs;
            inobs++;
        }
        //cout << " " << observations[i] ;
    }
    //cout << endl;

    MDPI_R mdpi(mdp, observations, initial);
    auto&& randompolicy = mdpi.random_policy(25);

    auto isol = mdpi.solve_reweighted(0, 0.9, randompolicy);
    BOOST_CHECK_EQUAL_COLLECTIONS(randompolicy.begin(), randompolicy.end(), isol.begin(), isol.end());
    isol = mdpi.solve_robust(0, 0.0, 0.9, randompolicy);
    BOOST_CHECK_EQUAL_COLLECTIONS(randompolicy.begin(), randompolicy.end(), isol.begin(), isol.end());

    isol = mdpi.solve_reweighted(1, 0.9, randompolicy);

    auto sol_impl = mpi_jac(*mdp, 0.9, numvec(0), PlainBellman(mdpi.obspol2statepol(isol)));

    BOOST_CHECK_CLOSE(sol_impl.total_return(initial), 51.3135, 0.1);
    BOOST_CHECK_CLOSE(mdpi.total_return(0.9), 51.3135, 0.1);

    isol = mdpi.solve_robust(1, 0.0, 0.9, randompolicy);
    sol_impl = mpi_jac(*mdp, 0.9, numvec(0), PlainBellman(mdpi.obspol2statepol(isol)));

    BOOST_CHECK_CLOSE(sol_impl.total_return(initial), 51.3135, 0.1);
    BOOST_CHECK_CLOSE(mdpi.total_return(0.9), 51.3135, 0.1);
}

#endif //  __cplusplus >= 201703L


BOOST_AUTO_TEST_CASE(test_return_of_implementable){
    // test return with different initial states

    const prec_t gamma = 0.99;

    MDP&& mdp = make_chain1();
    indvec observations = {0,0,0};
    Transition  initial1(numvec({1.0, 0.0, 0.0})),
                initial2(numvec({0.0, 1.0, 0.0})),
                initial3(numvec({0.0, 0.0, 1.0}));

    MDPI mdpi1(mdp, observations, initial1);
    BOOST_CHECK_CLOSE(mdpi1.total_return(gamma, 1e-5), 1.1*pow(gamma,2)/(1-gamma), 1e-3);
    MDPI mdpi2(mdp, observations, initial2);
    BOOST_CHECK_CLOSE(mdpi2.total_return(gamma, 1e-5), 1.1*pow(gamma,1)/(1-gamma), 1e-3);
    MDPI mdpi3(mdp, observations, initial3);
    BOOST_CHECK_CLOSE(mdpi3.total_return(gamma, 1e-5), 1.1*pow(gamma,0)/(1-gamma), 1e-3);
}


// TODO: make sure there is a test that checks that the return of the implementable policy with
// the true weights has the same return as the true MDP.


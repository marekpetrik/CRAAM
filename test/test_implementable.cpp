#include "RMDP.hpp"
#include "ImMDP.hpp"
#include "Transition.hpp"
#include "Simulation.hpp"
#include "Samples.hpp"

#include "cpp11-range-master/range.hpp"
#include <boost/functional/hash.hpp>
#include <iostream>
#include <iterator>
#include <cmath>

using namespace std;
using namespace craam;
using namespace craam::impl;
using namespace util::lang;


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


/**
Creates a simple chain problem.
Actions:   0 - left
           1 - right
Optimal solution: Action 1, with value function:
    [1.1 gamma^2/(1-gamma), 1.1 gamma/(1-gamma), 1.1/(1-gamma)]

*/
RMDP make_chain1(){
    RMDP rmdp(3);

    rmdp.add_transition_d(0,1,1,1,0);
    rmdp.add_transition_d(1,1,2,1,0);
    rmdp.add_transition_d(2,1,2,1,1.1);

    rmdp.add_transition_d(0,0,0,1,0);
    rmdp.add_transition_d(1,0,0,1,1);
    rmdp.add_transition_d(2,0,1,1,1);

    return rmdp;
}

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

    auto&& pol2 = imr.solve_robust(10, 0.0, 0.9);
    BOOST_CHECK_EQUAL_COLLECTIONS(pol2.begin(), pol2.end(),polvec.begin(),polvec.end());


    //auto retval = imr.total_return(pol, 0.99);
    //cout << "Return: " << retval << endl;

    //ostream_iterator<prec_t> output(cout, ", ");
    //copy(pol.begin(), pol.end(), output);
}

BOOST_AUTO_TEST_CASE(simple_mdpo_save_load_save_load) {
    RMDP&& rmdp1 = make_chain1();

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
    RMDP&& rmdp1 = make_chain1();

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

/**
A simple simulator class. The state represents a position in a chain
and actions move it up and down. The reward is equal to the position.

Representation
~~~~~~~~~~~~~~
- Decision state: position (int)
- Action: change (int)
- Expectation state: position, action (int,int)
*/
class Counter{
private:
    default_random_engine gen;
    bernoulli_distribution d;
    const vector<int> actions_list = {1,-1};
    const int initstate;

public:
    typedef int DState;
    typedef pair<int,int> EState;
    typedef int Action;

    /**
    Define the success of each action
    \param success The probability that the action is actually applied
    */
    Counter(double success, int initstate, random_device::result_type seed = random_device{}())
        : gen(seed), d(success), initstate(initstate) {};

    int init_state() const {
        return initstate;
    }

    pair<int,int> transition_dec(int state, int action) const{
        return make_pair(state,action);
    }

    pair<double,int> transition_exp(const pair<int,int> expstate) {
        int pos = expstate.first;
        int act = expstate.second;

        int nextpos = d(gen) ? pos + act : pos;
        return make_pair((double) pos, nextpos);
    }

    bool end_condition(const int state){
        return false;
    }

    vector<int> actions(int) const{
        return actions_list;
    }

    vector<int> actions() const{
        return actions_list;
    }
};

/** A counter that terminates at either end as defined by the end state */
class CounterTerminal : public Counter {
public:
    int endstate;

    CounterTerminal(double success, int initstate, int endstate, random_device::result_type seed = random_device{}())
        : Counter(success, initstate, seed), endstate(endstate) {};

    bool end_condition(const int state){
        return (abs(state) >= endstate);
    }
};
// Hash function for the Counter / CounterTerminal EState above
namespace std{
    template<> struct hash<pair<int,int>>{
        size_t operator()(pair<int,int> const& s) const{
            boost::hash<pair<int,int>> h;
            return h(s);
        };
    };
}

using namespace craam::msen;

/*
template<class T>
void print_vector(vector<T> vec){
    for(auto&& p : vec){
        cout << p << " ";
    }
}*/

BOOST_AUTO_TEST_CASE(implementable_from_samples){
    const int terminal_state = 10;

    CounterTerminal sim(0.9,0,terminal_state,1);
    RandomPolicySI<CounterTerminal> random_pol(sim,1);

    Samples<CounterTerminal> samples;
    simulate_stateless(sim,samples,random_pol,50,50);
    simulate_stateless(sim,samples,[](int){return 1;},10,20);
    simulate_stateless(sim,samples,[](int){return -1;},10,20);

    SampleDiscretizerSI<CounterTerminal> sd;
    // initialize action values
    sd.add_action(-1); sd.add_action(+1);
    //initialize state values
    for(auto i : range(-terminal_state,terminal_state)) sd.add_dstate(i);

    sd.add_samples(samples);

    BOOST_CHECK_EQUAL(samples.initial.size(), sd.get_discrete()->initial.size());
    BOOST_CHECK_EQUAL(samples.decsamples.size(), sd.get_discrete()->decsamples.size());
    BOOST_CHECK_EQUAL(samples.expsamples.size(), sd.get_discrete()->expsamples.size());

    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());
    auto mdp = smdp.get_mdp();
    auto&& initial = smdp.get_initial();

    auto&& sol = mdp->mpi_jac_ave(numvec(0),0.9);

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
    auto sol_impl = mdp->vi_jac_fix(numvec(0),0.9, mdpi.obspol2statepol(isol),
                    indvec(mdp->state_count(), 0));

    BOOST_CHECK_CLOSE(sol_impl.total_return(initial), 51.3135, 1e-3);
    BOOST_CHECK_CLOSE(mdpi.total_return(isol, 0.9), 51.3135, 1e-3);

    isol = mdpi.solve_robust(1, 0.0, 0.9, randompolicy);
    sol_impl = mdp->vi_jac_fix(numvec(0),0.9, mdpi.obspol2statepol(isol),
                    indvec(mdp->state_count(), 0));

    BOOST_CHECK_CLOSE(sol_impl.total_return(initial), 51.3135, 1e-3);
    BOOST_CHECK_CLOSE(mdpi.total_return(isol, 0.9), 51.3135, 1e-3);
}

BOOST_AUTO_TEST_CASE(test_return_of_implementable){
    // test return with different initial states


    const prec_t gamma = 0.99;

    RMDP&& mdp = make_chain1();
    indvec observations = {0,0,0};
    Transition  initial1(numvec({1.0, 0.0, 0.0})),
                initial2(numvec({0.0, 1.0, 0.0})),
                initial3(numvec({0.0, 0.0, 1.0}));

    MDPI mdpi1(mdp, observations, initial1);
    BOOST_CHECK_CLOSE(mdpi1.total_return(indvec(1,1),gamma, 1e-5), 1.1*pow(gamma,2)/(1-gamma), 1e-3);
    MDPI mdpi2(mdp, observations, initial2);
    BOOST_CHECK_CLOSE(mdpi2.total_return(indvec(1,1),gamma, 1e-5), 1.1*pow(gamma,1)/(1-gamma), 1e-3);
    MDPI mdpi3(mdp, observations, initial3);
    BOOST_CHECK_CLOSE(mdpi3.total_return(indvec(1,1),gamma, 1e-5), 1.1*pow(gamma,0)/(1-gamma), 1e-3);
}


// TODO: make sure there is a test that checks that the return of the implementable policy with
// the true weights has the same return as the true MDP.

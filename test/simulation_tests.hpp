#include "craam/Simulation.hpp"
#include "craam/modeltools.hpp"
#include "craam/RMDP.hpp"
#include "craam/algorithms/values.hpp"

#include <iostream>
#include <sstream>
#include <random>
#include <utility>
#include <functional>

#include <boost/functional/hash.hpp>

using namespace std;
using namespace craam;
using namespace craam::msen;
using namespace craam::algorithms;
using namespace util::lang;



struct TestState{
    int index;

    TestState(int i) : index(i){
    };
};

class TestSim {

public:

    typedef TestState State;
    typedef int Action;

    TestState init_state() const{
        return TestState(1);
    }

    pair<double,TestState> transition(TestState, int) const{
        return pair<double,TestState>(1.0,TestState(1));
    };

    bool end_condition(TestState) const{
        return false;
    };

    long action_count(TestState) const{return 1;}

    int action(TestState, long) const{return 1;}

};

int test_policy(TestState){
    return 0;
}

BOOST_AUTO_TEST_CASE(basic_simulation) {
    TestSim sim;

    auto samples = simulate<TestSim>(sim,test_policy,10,5);
    BOOST_CHECK_EQUAL(samples.size(), 50);
}



/**
A simple simulator class. The state represents a position in a chain
and actions move it up and down. The reward is equal to the position.

Representation
~~~~~~~~~~~~~~
- State: position (int)
- Action: change (int)
*/
class Counter{
private:
    default_random_engine gen;
    bernoulli_distribution d;
    const vector<int> actions_list;
    const int initstate;

public:
    using State = int;
    using Action = int;

    /**
    Define the success of each action
    \param success The probability that the action is actually applied
    */
    Counter(double success, int initstate, random_device::result_type seed = random_device{}())
        : gen(seed), d(success), actions_list({1,-1}), initstate(initstate) {};

    int init_state() const {
        return initstate;
    }

    pair<double,int> transition(int pos, int action) {
        int nextpos = d(gen) ? pos + action : pos;
        return make_pair((double) pos, nextpos);
    }

    bool end_condition(const int){
        return false;
    }

    int action(State , long index) const{
        return actions_list[index];
    }
    
    vector<int> get_valid_actions(State state){
        return actions_list;
    }

    size_t action_count(State) const{return actions_list.size();};
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


BOOST_AUTO_TEST_CASE(simulation_multiple_counter_si ) {
    Counter sim(0.9,0,1);

    RandomPolicy<Counter> random_pol(sim,1);
    auto samples = simulate(sim,random_pol,20,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), -3.51759102217019, 0.0001);

    samples = simulate(sim,random_pol,1,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 0, 0.0001);

    Counter sim2(0.9,3,1);
    samples = simulate(sim2,random_pol,1,20);
    BOOST_CHECK_CLOSE(samples.mean_return(0.9), 3, 0.0001);
}

BOOST_AUTO_TEST_CASE(simulation_multiple_counter_si_return ) {
    Counter sim(0.9,0,1);

    RandomPolicy<Counter> random_pol(sim,1); // sets the random seed
    auto samples_returns = simulate_return(sim,0.9,random_pol,20,20);
    auto v = samples_returns.second;
    auto meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, -3.51759102217019, 0.0001);

    samples_returns = simulate_return(sim,0.9,random_pol,1,20);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, 0, 0.0001);

    Counter sim2(0.9,3,1);
    samples_returns = simulate_return(sim2,0.9,random_pol,1,20);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, 3, 0.0001);

    // test termination probability equals to discount
    samples_returns = simulate_return(sim,1.0,DeterministicPolicy<Counter>(sim,indvec{0,1,1}),1000,100,0.1,1);
    v = samples_returns.second;
    meanreturn = accumulate(v.begin(), v.end(), 0.0) / prec_t(v.size());
    BOOST_CHECK_CLOSE(meanreturn, 4.73684, 10.0);

}
BOOST_AUTO_TEST_CASE(sampled_mdp_reward){
    // check that the reward is constructed correctly from samples
    DiscreteSamples samples;

    // relevant samples (transition to 1)
    samples.add_sample(0,0,1,1.0,3.0,0,0);
    samples.add_sample(0,0,1,2.0,2.0,0,0);
    samples.add_sample(0,0,1,3.0,1.0,0,0);
    // irrelevant samples (do not transition to 1)
    samples.add_sample(0,0,2,0.0,1.0,0,0);
    samples.add_sample(0,0,3,0.0,1.0,0,0);
    samples.add_sample(0,0,0,0.0,1.0,0,0);

    SampledMDP smdp;

    smdp.add_samples(samples);
    auto reward = (*smdp.get_mdp())[0][0][0].get_rewards()[1];
    //cout << (*smdp.get_mdp())[0][0][0].get_rewards()[1] << endl;

    BOOST_CHECK_CLOSE(reward, 1.666666, 1e-4);

    // check that the reward is constructed correctly from samples
    DiscreteSamples samples2;

    // relevant samples (transition to 1)
    samples2.add_sample(0,0,1,2.0,9.0,0,0);
    samples2.add_sample(0,0,1,4.0,6.0,0,0);
    samples2.add_sample(0,0,1,6.0,3.0,0,0);
    // irrelevant samples (do not transition to 1)
    samples2.add_sample(0,0,2,0.0,1.0,0,0);
    samples2.add_sample(0,0,3,0.0,1.0,0,0);
    samples2.add_sample(0,0,0,0.0,1.0,0,0);

    smdp.add_samples(samples2);
    //cout << (*smdp.get_mdp())[0][0][0].get_rewards()[1] << endl;
    reward = (*smdp.get_mdp())[0][0][0].get_rewards()[1];
    BOOST_CHECK_CLOSE(reward, 2.916666666666, 1e-4);
}

BOOST_AUTO_TEST_CASE(construct_mdp_from_samples_si_pol){

    CounterTerminal sim(0.9,0,10,1);
    RandomPolicy<CounterTerminal> random_pol(sim,1);

    auto samples = make_samples<CounterTerminal>();
    simulate(sim,samples,random_pol,50,50);
    simulate(sim,samples,[](int){return 1;},10,20);
    simulate(sim,samples,[](int){return -1;},10,20);

    SampleDiscretizerSD<typename CounterTerminal::State,
                        typename CounterTerminal::Action> sd;
    sd.add_samples(samples);

    BOOST_CHECK_EQUAL(samples.get_initial().size(), sd.get_discrete()->get_initial().size());
    BOOST_CHECK_EQUAL(samples.size(), sd.get_discrete()->size());


    SampledMDP smdp;
    smdp.add_samples(*sd.get_discrete());

    shared_ptr<const MDP> mdp = smdp.get_mdp();

    // check that the number of actions is correct (2)
    for(size_t i = 0; i < mdp->state_count(); i++){
        if(mdp->get_state(i).action_count() > 0)
            BOOST_CHECK_EQUAL(mdp->get_state(i).action_count(), 2);
    }

    auto&& sol = mpi_jac(*mdp,0.9);

    BOOST_CHECK_CLOSE(sol.total_return(smdp.get_initial()), 51.313973, 1e-3);
}


template<class Model>
Model create_test_mdp_sim(){
    Model rmdp(3);

    // nonrobust
    // action 1 is optimal, with transition matrix [[0,1,0],[0,0,1],[0,0,1]] and rewards [0,0,1.1]
    add_transition<Model>(rmdp,0,1,1,1.0,0.0);
    add_transition<Model>(rmdp,1,1,2,1.0,0.0);
    add_transition<Model>(rmdp,2,1,2,1.0,1.1);

    add_transition<Model>(rmdp,0,0,0,1.0,0.0);
    add_transition<Model>(rmdp,1,0,0,1.0,1.0);
    add_transition<Model>(rmdp,2,0,1,1.0,1.0);

    add_transition<Model>(rmdp,1,2,1,0.5,0.5);

    return rmdp;
}

BOOST_AUTO_TEST_CASE(simulate_mdp){

    shared_ptr<MDP> m = make_shared<MDP>();
    *m = create_test_mdp_sim<MDP>();

    Transition initial({0},{1.0});

    ModelSimulator ms(m, initial,13);
    ModelRandomPolicy rp(ms,10);

    auto samples = simulate(ms, rp, 1000, 5, -1, 0.0, 10);

    BOOST_CHECK_EQUAL(samples.size(), 49);
    //cout << "Number of samples " << samples.size() << endl;

    SampledMDP smdp;

    smdp.add_samples(samples);

    auto newmdp = smdp.get_mdp();

    auto solution1 = mpi_jac(*m, 0.9);
    auto solution2 = mpi_jac(*newmdp, 0.9);

    BOOST_CHECK_CLOSE(solution1.total_return(initial),8.90971,1e-3);
    //cout << "Return in original MDP " << solution1.total_return(initial) << endl;

    BOOST_CHECK_CLOSE(solution2.total_return(initial),8.90971,1e-3);
    //cout << "Return in sampled MDP " << solution2.total_return(initial) << endl;

    // need to remove the terminal state from the samples
    indvec policy = solution2.policy;
    policy.pop_back();

    //cout << "Computed policy " << policy << endl;
    indvec policytarget{1,1,1};
    BOOST_CHECK_EQUAL_COLLECTIONS(policy.begin(), policy.end(), policytarget.begin(), policytarget.end());
    auto solution3 = mpi_jac(*m, 0.9, numvec(0), PolicyDeterministic(policy));

    BOOST_CHECK_CLOSE(solution3.total_return(initial), 8.90916, 1e-2);
    //cout << "Return of sampled policy in the original MDP " << solution3.total_return(initial) << endl;

    ModelDeterministicPolicy dp(ms, policy);
    auto samples_policy = simulate(ms, dp, 1000, 5);

    BOOST_CHECK_CLOSE(samples_policy.mean_return(0.9), 8.91, 1e-3);
    //cout << "Return of sampled " << samples_policy.mean_return(0.9) << endl;

    ModelRandomizedPolicy rizedp(ms, {{0.5,0.5},{0.5,0.4,0.1},{0.5,0.5}},0);

    auto randomized_samples = simulate(ms, rizedp, 1000, 5, -1, 0.0, 10);

    BOOST_CHECK_CLOSE(randomized_samples.mean_return(0.9), 4.01147, 1e-3);
    //cout << "Return of randomized samples " << randomized_samples.mean_return(0.9) << endl;
}

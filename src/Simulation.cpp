
#include<random>
#include<memory>

#include "Simulation.hpp"


using namespace std;


/*
 * Signature of static methods required for the simulator
 *
 * DState init_state()
 * EState transition_dec(const DState&, const Action&)
 * pair<double,DState> transition_exp(const EState&)
 * bool end_condition(const DState&)
 * vector<Action> actions(const DState&)  // needed for a random policy and value function policy
 */

template<class DState,class Action,class EState,class Simulator,Action (*policy)(DState)>
unique_ptr<Samples<DState,Action,EState>>
simulate_stateless(long horizon,long runs,prec_t prob_term=0.0,long tran_limit=-1){
    /** \brief Runs the simulator and generates samples. A simulator with no state
     *
     * \param sim Simulator that holds the state of the process
     * \param horizon Number of steps
     * \param prob_term The probability of termination in each step
     * \return Samples
     */


    unique_ptr<Samples<DState,Action,EState>> samples(new Samples<DState,Action,EState>());

    long transitions = 0;

    // initialize random numbers when appropriate
    default_random_engine generator;
    uniform_real_distribution<double> distribution(0.0,1.0);


    for(auto run=0l; run < runs; run++){

        DState&& decstate = Simulator::init_state();
        samples->add_initial(decstate);

        for(auto step=0l; step < horizon; step++){
            if(Simulator::end_condition(decstate))
                break;
            if(tran_limit > 0 && transitions > tran_limit)
                break;

            Action&& action = policy(decstate);
            EState&& expstate = Simulator::transition_dec(decstate,action);


            samples->add_dec(DecSample<DState,Action,EState>
                                (decstate, action, expstate, step, run));

            auto&& rewardstate = Simulator::transition_exp(expstate);

            auto reward = rewardstate.first;
            decstate = rewardstate.second;

            samples->add_exp(ExpSample<DState,EState>(expstate, decstate, reward, 1.0, step, run));

            // test the termination probability only after at least one transition
            if(prob_term > 0.0){
                if( distribution(generator) <= prob_term)
                    break;
            }
            transitions++;
        };

        if(tran_limit > 0 && transitions > tran_limit)
            break;
    }

    return samples;
};

struct TestDState{
    int index;

    TestDState(int i){
        this->index = i;
    };
};

struct TestEState{
    int index;

    TestEState(int i){
        this->index = i;
    };
};


class TestSim {
private:
    TestSim();

public:

    static TestDState init_state(){
        return TestDState(1);
    }

    static TestEState transition_dec(const TestDState&, const int&){
        return TestEState(1);
    };

    static pair<double,TestDState> transition_exp(const TestEState&){
        return pair<double,TestDState>(1.0,TestDState(1));
    };

    static bool end_condition(const TestDState&){
        return false;
    };

    static vector<int> actions(const TestDState&){
        return vector<int>{1};
    };

};

int test_policy(TestDState){
    return 0;
}

void test_compile(){
    simulate_stateless<TestDState,int,TestEState,TestSim,test_policy>(10,5);
};

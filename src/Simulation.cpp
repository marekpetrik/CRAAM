#include "Simulation.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace craam{
namespace msen {

//ModelSimulator::ModelSimulator(){
//}

auto ModelSimulator::init_state() -> State{
    
    const numvec& probs = initial.get_probabilities();
    const indvec& inds = initial.get_indices();

    auto dst = discrete_distribution<long>(probs.begin(), probs.end());

    return inds[dst(gen)];
}

auto  ModelSimulator::transition(State state, Action action) -> pair<double,State> {

    assert(state >= 0 && size_t(state) < mdp->size());
    const auto& mdpstate = (*mdp)[state];

    assert(action >= 0 && size_t(action) < mdpstate.size());
    const auto& mdpaction = mdpstate[action];
    const auto& tran = mdpaction.get_outcome();

    const numvec& probs = tran.get_probabilities();
    const numvec& rews = tran.get_rewards();
    const indvec& inds = tran.get_indices();

    // check if the transition sums to 1, if not use the remainder 
    // as a probability of terminating
    prec_t prob_termination = 1 - tran.sum_probabilities();

    discrete_distribution<long> dst;

    if(prob_termination > SOLPREC){
        // copy the probabilities (there should be a faster way too)
        numvec copy_probs(probs);
        copy_probs.push_back(prob_termination);

        dst = discrete_distribution<long>(copy_probs.begin(), copy_probs.end());
    }else{
        dst = discrete_distribution<long>(probs.begin(), probs.end());
    }

    const size_t nextindex = dst(gen);

    // check if need to transition to a terminal state
    const State nextstate = nextindex < inds.size() ? 
                            inds[nextindex] : mdp->size();

    // reward is zero when transitioning to the terminal state
    const prec_t reward = nextindex < inds.size() ? 
                            rews[nextindex] : 0.0;

    return make_pair(reward, nextstate);
}

}
}



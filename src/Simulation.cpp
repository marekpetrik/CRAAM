#include "Simulation.hpp"

#include <algorithm>
#include <cmath>
#include <string>

namespace craam{
namespace msen {


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

    auto dst = discrete_distribution<long>(probs.begin(), probs.end());

    const long nextindex = dst(gen);
    const State nextstate = inds[nextindex];
    const prec_t reward = rews[nextindex];

    return make_pair(reward, nextstate);
}

}
}



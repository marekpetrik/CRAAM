#include "Samples.hpp"
#include "modeltools.hpp"

#include <utility>
#include <vector>
#include <string>


namespace craam{namespace msen {

using namespace std;
using namespace util::lang;

SampledMDP::SampledMDP() : mdp(make_shared<MDP>()) {}

void SampledMDP::add_samples(const DiscreteSamples& samples){

    // copy the state and action counts to be
    vector<vector<size_t>> old_state_action_counts = state_action_counts;

    // add transition samples
    for(size_t si : indices(samples)){

        DiscreteSample s = samples.get_sample(si);

        // -----------------
        // Computes sample weight:
        // the idea is to normalize new samples by the same
        // value as the existing samples and then re-normalize
        // this is linear complexity
        // -----------------
        // this needs to be initialized to 1.0
        prec_t weight = 1.0;
        bool weight_initialized = false;

        // resize transition counts
        // the actual values are updated later
        if((size_t) s.state_from() >= state_action_counts.size()){
            state_action_counts.resize(s.state_from()+1);

            // we know that the value will not be found in old data
            weight_initialized = true;
        }

        // check if we have something for the action
        vector<size_t>& actioncount = state_action_counts[s.state_from()];
        if((size_t)s.action() >= actioncount.size()){
            actioncount.resize(s.action()+1);

            // we know that the value will not be found in old data
            weight_initialized = true;
        }

        // update the new count
        assert(size_t(s.state_from()) < state_action_counts.size());
        assert(size_t(s.action()) < state_action_counts[s.state_from()].size());

        state_action_counts[s.state_from()][s.action()]++;

        // get number of existing transitions
        // this is only run when we do not know that we have no prior
        // sample
        if(!weight_initialized &&
                (size_t(s.state_from()) < old_state_action_counts.size()) &&
                (size_t(s.action()) < old_state_action_counts[s.state_from()].size())) {

            size_t cnt = old_state_action_counts[s.state_from()][s.action()];

            // adjust the weight of the new sample to be consistent
            // with the previous normalization (use 1.0 if no previous action)
            weight = 1.0 / prec_t(cnt);
        }
        // -----------------------

        // adds a transition
        add_transition( *mdp, s.state_from(), s.action(), s.state_to(),
                        weight*s.weight(),
                        s.reward());
    }

    // make sure that there are no actions with no samples
    for(size_t si : indices(*mdp)){
        const auto& state = mdp->get_state(si);
        for(size_t ai : indices(state)){
            if(state.get_action(ai).get_outcome().empty())
                throw invalid_argument("No sample for state " + to_string(si) + " and action " + to_string(ai) + ".");
        }
    }

    //  Normalize the transition probabilities
    mdp->normalize();

    // set initial distribution
    for(long state : samples.get_initial()){
        initial.add_sample(state, 1.0, 0.0);
    }
    initial.normalize();
}
}}

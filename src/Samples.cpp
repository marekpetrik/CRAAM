#include "Samples.hpp"

#include <utility>
#include <vector>
#include <string>

using namespace std;

using namespace util::lang;

namespace craam{
namespace msen {

SampledMDP::SampledMDP() : mdp(make_shared<MDP>()) {}

void SampledMDP::add_samples(const DiscreteSamples& samples){

    if(initialized)
        throw invalid_argument("Multiple calls not supported yet.");

    // ** For each expectation state index, save the state and action number
    // maps expectation state numbers to decision state number and action it comes from
    vector<vector<pair<long,long>>> expstate2da(0);
    for(const DiscreteDSample& ds : samples.decsamples){
        auto esid = ds.expstate_to;
        // resize if necessary
        if(esid >= (long) expstate2da.size()){
            expstate2da.resize(esid+1, vector<pair<long,long>>(0));
        }
        expstate2da[esid].push_back(make_pair(ds.decstate_from, ds.action));
    }

    // ** Go by expectation states and determine where to add the sample
    for(const DiscreteESample& es : samples.expsamples){

        for(const auto sa : expstate2da[es.expstate_from]){
            long decstate = sa.first;
            long action = sa.second;

            // make sure that the destination state exists
            mdp->create_state(es.decstate_to);

            mdp->create_state(decstate).create_action(action).get_outcome().
                    add_sample(es.decstate_to, es.weight, es.reward);
        }
    }

    // make sure that there are no missing action samples
    for(size_t si : indices(*mdp)){
        const auto& state = mdp->get_state(si);
        for(size_t ai : indices(state)){
            if(state.get_action(ai).get_outcome().empty()){
                throw invalid_argument("No sample for state " + to_string(si) + " and action " + to_string(ai) + ".");
            }
        }
    }

    // ** Then normalize the transition probabilities
    mdp->normalize();

    // set initial distribution
    for(long state : samples.initial){
        initial.add_sample(state, 1.0, 0.0);
    }
    initial.normalize();
    initialized = true;
}
}
}

#include "Samples.hpp"

#include <utility>
#include <vector>
#include <string>

using namespace std;

namespace craam{
namespace msen {

SampledMDP::SampledMDP() : mdp(make_shared<RMDP>()) {}

void SampledMDP::copy_samples(const DiscreteSamples& samples){

    if(initialized)
        throw invalid_argument("Multiple calls not supported yet.");

    // ** For each expectation state index, save the state and action number
    // a single decision state and an action should not lead to different
    // expectation states

    // maps expectation state numbers to decision state number and action
    vector<pair<long,long>> expstate2da(0);

    for(const DiscreteDSample& ds : samples.decsamples){
        auto esid = ds.expstate_to;

        // resize if necessary
        if(esid >= (long) expstate2da.size()){
            expstate2da.resize(esid+1, make_pair(-1,-1));
        }

        if(expstate2da[esid].first >= 0)
            throw invalid_argument("Non-unique expectation state numbers for the same state and action pair: " + to_string(ds.decstate_from) + "," + to_string(ds.action));

        expstate2da[esid] = make_pair(ds.decstate_from, ds.action);
    }

    // ** Go by expectation states and determine where to add the sample
    for(const DiscreteESample& es : samples.expsamples){
        long decstate = expstate2da[es.expstate_from].first;
        long action = expstate2da[es.expstate_from].second;

        mdp->assure_state_exists(es.decstate_to);
        Transition& t = mdp->create_transition(decstate, action, 0);

        t.add_sample(es.decstate_to, es.weight, es.reward);
    }

    // ** Then normalize the transitions
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

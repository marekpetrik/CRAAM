#include "ImMDP.hpp"

#include "definitions.hpp"
#include <algorithm>
#include <memory>

using namespace std;

namespace craam{
namespace impl{


unique_ptr<RMDP> IMDP::to_robust(){
    /**
        Constructs a robust version of the implementable MDP.
    */

    // *** check consistency of provided parameters ***
    // check that the number of observations coefficients it correct
    if(mdp->state_count() != observations.size())
        throw invalid_argument("Number of observation indexes must match the number of states.");
    // check that the observation indices are not negative
    if(*min_element(observations.begin(), observations.end()) < 0)
        throw invalid_argument("Observation indices must be non-negative");

    // *** will check the following properties in the code
    // check that there is no robustness
    // make sure that the action sets for each observation are the same

    // Determine the number of observations
    auto obs_count = *max_element(observations.begin(), observations.end()) + 1;

    unique_ptr<RMDP> result(new RMDP(obs_count));

    // keep track of the number of outcomes for each
    vector<long> outcome_count(obs_count, 0);
    // keep track of which outcome a state is mapped to
    vector<long> outcome_id(obs_count, -1);

    for(size_t state_index=0; state_index < mdp->state_count(); state_index++){

    }

    return result;
}

}}


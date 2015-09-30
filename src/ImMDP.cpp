#include "ImMDP.hpp"

#include "definitions.hpp"
#include <algorithm>
#include <memory>

using namespace std;

namespace craam{
namespace impl{


MDPI::MDPI(const shared_ptr<const RMDP>& mdp, const vector<long>& observ2state, const Transition& initial)
            : mdp(mdp), observ2state(observ2state), initial(initial){
    /**
        Constructs the MDP with implementability constraints.

        \param mdp A non-robust base MDP model. It cannot be shared to prevent
                    direct modification.
        \param observ2state Maps each state to the index of the corresponding observation.
                        A valid policy will take the same action in all states
                        with a single observation. The index is 0-based.
        \param initial A representation of the initial distribution. The rewards
                        in this transition are ignored (and should be 0).
    */

    // *** check consistency of provided parameters ***
    // check that the number of observ2state coefficients it correct
    if(mdp->state_count() != (long) observ2state.size())
        throw invalid_argument("Number of observation indexes must match the number of states.");
    // check that the observation indices are not negative
    if(*min_element(observ2state.begin(), observ2state.end()) < 0)
        throw invalid_argument("Observation indices must be non-negative");
    // check then initial transition
    if(initial.max_index() >= mdp->state_count())
        throw invalid_argument("An initial transition to a non-existent state.");
    if(!initial.is_normalized())
        throw invalid_argument("The initial transition must be normalized.");

}

MDPI_R::MDPI_R(const shared_ptr<const RMDP>& mdp, const vector<long>& observ2state,
            const Transition& initial) : MDPI(mdp, observ2state, initial){

    initialize_robustmdp();
}

void MDPI_R::initialize_robustmdp(){
    /**
        Constructs a robust version of the implementable MDP.
    */
    // *** will check the following properties in the code
    // check that there is no robustness
    // make sure that the action sets for each observation are the same

    // Determine the number of observ2state
    auto obs_count = *max_element(observ2state.begin(), observ2state.end()) + 1;


    // keep track of the number of outcomes for each
    vector<long> outcome_count(obs_count, 0);
    // keep track of which outcome a state is mapped to
    vector<long> state2outcome(mdp->state_count(), -1);
    // keep track of actions - needs to make sure that they are all the same
    vector<long> action_counts(obs_count, -1);  // -1 means not initialized

    for(size_t state_index=0; (long) state_index < mdp->state_count(); state_index++){
        auto obs = observ2state[state_index];

        // check the number of actions
        auto ac = mdp->action_count(state_index);
        if(action_counts[obs] >= 0){
            if(action_counts[obs] != ac){
                throw invalid_argument("Inconsistent number of actions: " + to_string(state_index) +
                                       " instead of " + to_string(action_counts[obs]) +
                                       " in state " + to_string(state_index));}
        }else{
            action_counts[obs] = ac;
        }

        // maps the transitions
        for(size_t action_index=0; (long) action_index < ac; action_index++){
            // check to make sure that there is no robustness
            if(mdp->outcome_count(state_index,action_index) > 1)
                throw invalid_argument("Robust base MDP is not supported; multiple outcomes in state " +
                                       to_string(state_index) + " and action " + to_string(action_index) );

            const Transition& old_tran = mdp->get_transition(state_index,action_index, 0);
            Transition& new_tran = robust_mdp.get_transition(obs,action_index,outcome_count[obs]);

            // copy the original transitions (they are automatically consolidated while being added)
            for(size_t k=0; k< old_tran.size(); k++){

                new_tran.add_sample(observ2state[old_tran.indices[k]],
                                    old_tran.probabilities[k],
                                    old_tran.rewards[k]);
            }

        }
        state2outcome[state_index] = outcome_count[obs]++;
    }
}

}}


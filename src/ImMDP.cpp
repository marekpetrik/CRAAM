#include "ImMDP.hpp"

#include "definitions.hpp"
#include <algorithm>
#include <memory>

#include <iostream>

using namespace std;

namespace craam{
namespace impl{


void MDPI::check_parameters(const RMDP& mdp, const vector<long>& state2observ, const Transition& initial){
    /**
         Checks whether the parameters are correct. Throws an exception if the parmaters
         are wrong.
     */

    // *** check consistency of provided parameters ***
    // check that the number of state2observ coefficients it correct
    if(mdp.state_count() !=  state2observ.size())
        throw invalid_argument("Number of observation indexes must match the number of states.");
    // check that the observation indices are not negative
    if(*min_element(state2observ.begin(), state2observ.end()) < 0)
        throw invalid_argument("Observation indices must be non-negative");
    // check then initial transition
    if(initial.max_index() >= (long) mdp.state_count())
        throw invalid_argument("An initial transition to a non-existent state.");
    if(!initial.is_normalized())
        throw invalid_argument("The initial transition must be normalized.");

}


MDPI::MDPI(const shared_ptr<const RMDP>& mdp, const vector<long>& state2observ, const Transition& initial)
            : mdp(mdp), state2observ(state2observ), initial(initial){
    /**
        Constructs the MDP with implementability constraints. This constructor makes it 
        possible to share the MDP with other data structures.

        \param mdp A non-robust base MDP model. It cannot be shared to prevent
                    direct modification.
        \param state2observ Maps each state to the index of the corresponding observation.
                        A valid policy will take the same action in all states
                        with a single observation. The index is 0-based.
        \param initial A representation of the initial distribution. The rewards
                        in this transition are ignored (and should be 0).
    */

    check_parameters(*mdp, state2observ, initial);
}

MDPI::MDPI(const RMDP& mdp, const vector<long>& state2observ, const Transition& initial)
            : mdp(new RMDP(mdp)), 
            state2observ(state2observ), initial(initial){
    /**
        Constructs the MDP with implementability constraints. The MDP model is
        copied (using the copy constructor) and stored internally.

        \param mdp A non-robust base MDP model. It cannot be shared to prevent
                    direct modification.
        \param state2observ Maps each state to the index of the corresponding observation.
                        A valid policy will take the same action in all states
                        with a single observation. The index is 0-based.
        \param initial A representation of the initial distribution. The rewards
                        in this transition are ignored (and should be 0).
    */

    check_parameters(mdp, state2observ, initial);
}

MDPI_R::MDPI_R(const shared_ptr<const RMDP>& mdp, const vector<long>& state2observ,
            const Transition& initial) : MDPI(mdp, state2observ, initial){
    /**
        Calls the base constructor and also constructs the corresponding
        robust MDP
     */

    initialize_robustmdp();
}

MDPI_R::MDPI_R(const RMDP& mdp, const vector<long>& state2observ,
            const Transition& initial) : MDPI(mdp, state2observ, initial){
    /**
        Calls the base constructor and also constructs the corresponding
        robust MDP
     */

    initialize_robustmdp();
}


void MDPI_R::initialize_robustmdp(){
    /**
        Constructs a robust version of the implementable MDP.
    */
    // *** will check the following properties in the code
    // check that there is no robustness
    // make sure that the action sets for each observation are the same

    // Determine the number of state2observ
    auto obs_count = *max_element(state2observ.begin(), state2observ.end()) + 1;


    // keep track of the number of outcomes for each
    vector<long> outcome_count(obs_count, 0);
    // keep track of which outcome a state is mapped to
    vector<long> state2outcome(mdp->state_count(), -1);
    // keep track of actions - needs to make sure that they are all the same
    vector<long> action_counts(obs_count, -1);  // -1 means not initialized

    for(size_t state_index=0; state_index < mdp->state_count(); state_index++){
        auto obs = state2observ[state_index];

        // check the number of actions
        auto ac = mdp->action_count(state_index);
        if(action_counts[obs] >= 0){
            if(action_counts[obs] != (long) ac){
                throw invalid_argument("Inconsistent number of actions: " + to_string(state_index) +
                                       " instead of " + to_string(action_counts[obs]) +
                                       " in state " + to_string(state_index));}
        }else{
            action_counts[obs] = ac;
        }

        // maps the transitions
        for(long action_index=0; action_index < (long) ac; action_index++){
            // check to make sure that there is no robustness
            if(mdp->outcome_count(state_index,action_index) > 1)
                throw invalid_argument("Robust base MDP is not supported; multiple outcomes in state " +
                                       to_string(state_index) + " and action " + to_string(action_index) );

            const Transition& old_tran = mdp->get_transition(state_index,action_index, 0);
            Transition& new_tran = robust_mdp.get_transition(obs,action_index,outcome_count[obs]);

            // copy the original transitions (they are automatically consolidated while being added)
            for(size_t k=0; k< old_tran.size(); k++){

                new_tran.add_sample(state2observ[old_tran.get_indices()[k]],
                                    old_tran.get_probabilities()[k],
                                    old_tran.get_rewards()[k]);
            }

        }
        state2outcome[state_index] = outcome_count[obs]++;
    }
}

}}


#include "ImMDP.hpp"

#include "definitions.hpp"
#include <algorithm>
#include <memory>

#include <iostream>
#include <iterator>

#include "cpp11-range-master/range.hpp"

using namespace std;
using namespace util::lang;


namespace craam{
namespace impl{


void MDPI::check_parameters(const RMDP& mdp, const indvec& state2observ,
                            const Transition& initial){
    /**
         Checks whether the parameters are correct. Throws an exception if the parmaters
         are wrong.
     */

    // *** check consistency of provided parameters ***
    // check that the number of state2observ coefficients it correct
    if(mdp.state_count() !=  state2observ.size())
        throw invalid_argument("Number of observation indexes must match the number of states.");
    // check that the observation indexes are not negative
    if(*min_element(state2observ.begin(), state2observ.end()) < 0)
        throw invalid_argument("Observation indexes must be non-negative");
    // check then initial transition
    if(initial.max_index() >= (long) mdp.state_count())
        throw invalid_argument("An initial transition to a non-existent state.");
    if(!initial.is_normalized())
        throw invalid_argument("The initial transition must be normalized.");
}

MDPI::MDPI(const shared_ptr<const RMDP>& mdp, const indvec& state2observ,
           const Transition& initial)
            : mdp(mdp), state2observ(state2observ), initial(initial),
              obscount(1+*max_element(state2observ.begin(), state2observ.end())){
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

MDPI::MDPI(const RMDP& mdp, const indvec& state2observ, const Transition& initial)
            : MDPI(shared_ptr<const RMDP>(new RMDP(mdp)),state2observ, initial){
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
}

MDPI_R::MDPI_R(const shared_ptr<const RMDP>& mdp, const indvec& state2observ,
            const Transition& initial) : MDPI(mdp, state2observ, initial){
    /**
        Calls the base constructor and also constructs the corresponding
        robust MDP
     */

    initialize_robustmdp();
}

indvec MDPI::obspol2statepol(indvec obspol) const{
    /**
        Converts a policy defined in terms of observations to a policy defined in
        terms of states.

        \param obspol Policy that maps observations to actions to take
     */

     indvec statepol(state_count());

     for(size_t s=0; s < state_count(); s++){
         statepol[s] = obspol[state2observ[s]];
     }

     return statepol;
}

void MDPI::to_csv(ostream& output_mdp, ostream& output_state2obs,
                  ostream& output_initial, bool headers) const{
    /**
    Saves the MDPI to a set of 3 csv files, for transitions,
    observations, and the initial distribution

    \param output_mdp Transition probabilities and rewards
    \param output_state2obs Mapping states to observations
    \param output_initial Initial distribution
    */

    // save the MDP
    mdp->to_csv(output_mdp, headers);

    // save state maps
    if(headers){
        output_state2obs << "idstate,idobs" << endl;
    }
    for(auto i : indices(state2observ)){
        output_state2obs << i << "," << state2observ[i] << endl;
    }

    // save the initial distribution
    if(headers){
        output_initial << "idstate,probability" << endl;
    }
    const indvec& inindices = initial.get_indices();
    const numvec& probabilities = initial.get_probabilities();

    for(auto i : indices(inindices)){
        output_initial << inindices[i] << "," << probabilities[i];
    }

}

void MDPI::to_csv_file(const string& output_mdp, const string& output_state2obs,
                       const string& output_initial, bool headers) const{
    /**
    Saves the MDPI to a set of 3 csv files, for transitions, observations,
    and the initial distribution

    \param output_mdp File name for transition probabilities and rewards
    \param output_state2obs File name for mapping states to observations
    \param output_initial File name for initial distribution
    */

    // open file streams
    ofstream ofs_mdp(output_mdp),
                ofs_state2obs(output_state2obs),
                ofs_initial(output_initial);

    // save the data
    to_csv(ofs_mdp, ofs_state2obs, ofs_initial, headers);

    // close streams
    ofs_mdp.close(); ofs_state2obs.close(); ofs_initial.close();
}

unique_ptr<MDPI> MDPI::from_csv(istream& input_mdp, istream& input_state2obs,
                                istream& input_initial, bool headers){
    /**
    Loads an MDPI from a set of 3 csv files, for transitions, observations,
    and the initial distribution

    The MDP size is defined by the transitions file.

    \param input_mdp File name for transition probabilities and rewards
    \param input_state2obs File name for mapping states to observations
    \param input_initial File name for initial distribution
     */

    // read mdp
    auto mdp = RMDP::from_csv(input_mdp);

    // read state2obs
    string line;
    if(headers) input_state2obs >> line; // skip the header

    indvec state2obs(mdp->state_count());
    input_state2obs >> line;
    while(input_state2obs.good()){
        string cellstring;
        stringstream linestream(line);

        getline(linestream, cellstring, ',');
        auto idstate = stoi(cellstring);
        getline(linestream, cellstring, ',');
        auto idobs = stoi(cellstring);
        state2obs[idstate] = idobs;

        input_state2obs >> line;
    }

    // read initial distribution
    if(headers) input_initial >> line; // skip the header

    Transition initial;
    input_initial >> line;
    while(input_initial.good()){
        string cellstring;
        stringstream linestream(line);

        getline(linestream, cellstring, ',');
        auto idstate = stoi(cellstring);
        getline(linestream, cellstring, ',');
        auto prob = stof(cellstring);
        initial.add_sample(idstate, prob, 0.0);

        input_initial >> line;
    }

    shared_ptr<const RMDP> csmdp = const_pointer_cast<const RMDP>(
                            shared_ptr<RMDP>(std::move(mdp)));
    return make_unique<MDPI>(csmdp, state2obs, initial);
}

unique_ptr<MDPI> MDPI::from_csv_file(const string& input_mdp, const string& input_state2obs,
                                     const string& input_initial, bool headers){

    // open files
    ifstream ifs_mdp(input_mdp),
                ifs_state2obs(input_state2obs),
                ifs_initial(input_initial);

    // transfer method call
    return from_csv(ifs_mdp, ifs_state2obs, ifs_initial, headers);
}

MDPI_R::MDPI_R(const RMDP& mdp, const indvec& state2observ,
            const Transition& initial) : MDPI(mdp, state2observ, initial){
    /**
        Calls the base constructor and also constructs the corresponding
        robust MDP.
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
    indvec outcome_count(obs_count, 0);
    // keep track of which outcome a state is mapped to
    indvec state2outcome(mdp->state_count(), -1);
    // keep track of actions - needs to make sure that they are all the same
    indvec action_counts(obs_count, -1);  // -1 means not initialized

    for(size_t state_index=0; state_index < mdp->state_count(); state_index++){
        auto obs = state2observ[state_index];

        // check the number of actions
        auto ac = mdp->get_state(state_index).action_count();
        if(action_counts[obs] >= 0){
            if(action_counts[obs] != (long) ac){
                throw invalid_argument("Inconsistent number of actions: " + to_string(state_index) +
                                       " instead of " + to_string(action_counts[obs]) +
                                       " in state " + to_string(state_index));
            }
        }else{
            action_counts[obs] = ac;
        }

        // maps the transitions
        for(long action_index=0; action_index < (long) ac; action_index++){
            // check to make sure that there is no robustness
            if(mdp->get_state(state_index).get_action(action_index).outcome_count() > 1)
                throw invalid_argument("Robust base MDP is not supported; multiple outcomes in state " +
                                       to_string(state_index) + " and action " + to_string(action_index) );

            const Transition& old_tran = mdp->get_transition(state_index,action_index,0);
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

void MDPI_R::update_importance_weights(const numvec& weights){
    /**
        Updates the weights on outcomes in the robust MDP based on the state
        weights provided.

        This method modifies the stored robust MDP.
     */

    if(weights.size() != state_count()){
        throw invalid_argument("Size of distribution must match the number of states.");
    }

    // loop over all mdp states and set weights
    for(size_t i = 0; i < weights.size(); i++){
        const auto rmdp_stateid = state2observ[i];
        const auto rmdp_outcomeid = state2outcome[i];
        // loop over all actions

        auto& rstate = robust_mdp.get_state(rmdp_stateid);
        for(auto& a : rstate.actions){
            a.set_distribution(rmdp_outcomeid, weights[i]);
        }
    }

    // now normalize the weights to they sum to one
    for(auto& s : robust_mdp.states){
        for(auto& a : s.actions){
            a.normalize_distribution();
        }
    }
}

indvec MDPI_R::solve_reweighted(long iterations, prec_t discount){
    /**
        Uses a simple iterative algorithm to solve the MDPI.

        The algorithm starts with a policy composed of actions all 0, and
        then updates the distribution of robust outcomes (corresponding to MDP states),
        and computes the optimal solution for thus weighted RMDP.

        This method modifies the stored robust MDP.

        \param iterations Maximal number of iterations;
                    also stops if the policy no longer changes

        \returns Policy for observations (an index of each action for each observation)
     */

   // TODO: add a method in RMDP to compute the distribution of a non-robust policy
    const indvec nature(state_count(), 0);

    indvec obspol_ret(0);         // current policy in terms of observations
    indvec statepol(state_count(), 0); // state policy that corresponds to the observation policy

    for(long iter=0; iter < iterations; iter++){

        // compute state distribution
        auto&& importanceweights = mdp->ofreq_mat(initial, discount, statepol, nature);

        // update importance weights
        update_importance_weights(importanceweights);

        // compute solution of the robust MDP with the new weights
        Solution&& s = robust_mdp.mpi_jac_ave(numvec(0),discount,10000,0.1,10000,0.1);

        // update the policy for the underlying states
        auto&& obspol = s.policy;

        // map the observation policy to the individual states
        for(size_t statei=0; statei < state_count(); statei++){
            statepol[statei] = obspol[state2observ[statei]];
        }

        // update the return value in the last iteration
        if(iter == iterations-1){
            obspol_ret = obspol;
        }

    }
    return obspol_ret;
}

}}


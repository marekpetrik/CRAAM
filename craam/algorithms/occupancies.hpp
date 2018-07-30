#pragma once

#include "craam/RMDP.hpp"

#include <Eigen/Dense>
#include <rm/range.hpp>

namespace craam{namespace algorithms{

using namespace std;
using namespace Eigen;

/// Internal helper functions
namespace internal{

    /// Helper function to deal with variable indexing
    template<class SType>
    inline Transition mean_transition_state(const SType& state, long index, const pair<indvec,vector<numvec>>& policies){
        return state.mean_transition(policies.first[index], policies.second[index]);
    }

    /// Helper function to deal with variable indexing
    template<class SType>
    inline Transition mean_transition_state(const SType& state, long index, const indvec& policy){
        return state.mean_transition(policy[index]);
    }

    /// Helper function to deal with variable indexing
    template<class SType>
    inline prec_t mean_reward_state(const SType& state, long index, const pair<indvec,vector<numvec>>& policies){
        return state.mean_reward(policies.first[index], policies.second[index]);
    }

    /// Helper function to deal with variable indexing
    template<class SType>
    inline prec_t mean_reward_state(const SType& state, long index, const indvec& policy){
        return state.mean_reward(policy[index]);
    }
}

/**
Constructs the transition (or its transpose) matrix for the policy.

\tparam SType Type of the state in the MDP (regular vs robust)
\tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic 
                policy and a randomized policy of the nature
\param rmdp Regular or robust MDP
\param policies The policy (indvec) or the pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically 
        a randomized policy
\param transpose (optional, false) Whether to return the transpose of the transition matrix. 
        This is useful for computing occupancy frequencies
*/
template<typename SType, typename Policies> 
inline MatrixXd transition_mat(const GRMDP<SType>& rmdp, const Policies& policies, bool transpose = false) {
    const size_t n = rmdp.state_count();
    MatrixXd result = MatrixXd::Zero(n,n);

    const auto& states = rmdp.get_states();
    #pragma omp parallel for
    for(size_t s = 0; s < n; s++){
        const Transition&& t = internal::mean_transition_state(states[s], s, policies);

        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        if(!transpose){
            for(size_t j=0; j < t.size(); j++)
                result(s,indexes[j]) = probabilities[j];
        }else{
            for(size_t j=0; j < t.size(); j++)
                result(indexes[j],s) = probabilities[j];
        }
    }
    return result;
}

/**
Constructs the rewards vector for each state for the RMDP.

\tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic 
                policy and a randomized policy of the nature
\param rmdp Regular or robust MDP
\param policies The policy (indvec) or the pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically 
        a randomized policy
 */
template<typename SType, typename Policy>
inline numvec rewards_vec(const GRMDP<SType>& rmdp, const Policy& policies){
    
    const auto n = rmdp.state_count();
    numvec rewards(n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        const SType& state = rmdp[s];
        if(state.is_terminal())
            rewards[s] = 0;
        else
            rewards[s] = internal::mean_reward_state(state, s, policies);
    }
    return rewards;
}

/**
Computes occupancy frequencies using matrix representation of transition
probabilities. This method may not scale well


\tparam SType Type of the state in the MDP (regular vs robust)
\tparam Policy Type of the policy. Either a single policy for
                the standard MDP evaluation, or a pair of a deterministic 
                policy and a randomized policy of the nature
\param init Initial distribution (alpha)
\param discount Discount factor (gamma)
\param policies The policy (indvec) or the pair of the policy and the policy
        of nature (pair<indvec,vector<numvec> >). The nature is typically 
        a randomized policy
*/
template<typename SType, typename Policies>
inline numvec 
occfreq_mat(const GRMDP<SType>& rmdp, const Transition& init, prec_t discount,
                 const Policies& policies) {
    const auto n = rmdp.state_count();

    // initial distribution
    const numvec& ivec = init.probabilities_vector(n);
    const VectorXd initial_vec = Map<const VectorXd,Unaligned>(ivec.data(),ivec.size());

    // get transition matrix and construct (I - gamma * P^T)
    MatrixXd t_mat = MatrixXd::Identity(n,n)  - discount * transition_mat(rmdp, policies, true);

    // solve set of linear equations
    numvec result(n,0);
    Map<VectorXd,Unaligned>(result.data(),result.size()) = HouseholderQR<MatrixXd>(t_mat).solve(initial_vec);

    return result;
}

/**
Computes occupancy with a given horizon.  Guaranteed to not scale well.
*/
template<typename SType>
inline prob_matrix_t
occfreq_action_horizon_stochcastic
(const GRMDP<SType>& rmdp, const Transition& init, prec_t discount,
                 const prob_matrix_t& policy, int horizon) {
    const auto state_count = rmdp.state_count();
    const auto action_count = rmdp.action_count();

    //Create the inital occupancy matrix given the initial state distribution
    prob_matrix_t occupancy_matrix(state_count, prob_list_t(action_count,0));
    const numvec& ivec = init.probabilities_vector(state_count);
    for ( int current_state = 0; current_state < state_count; current_state++ ){
        const prob_list_t &policy_current_state = policy[current_state];
        for ( int current_action = 0; current_action < action_count; current_action++ )
            occupancy_matrix[current_state][current_action] = ivec[current_state] * policy_current_state[current_action];
    }

    for ( int t = 1; t < horizon; t++ ) {
        prob_matrix_t *new_occupancy_additions = new prob_matrix_t(state_count, prob_list_t(action_count,0));

        for ( int current_state = 0; current_state < state_count; current_state++ ){
            prec_t occupancy_of_current_state = 0;
            for ( int current_action = 0; current_action < action_count; current_action++ )
                occupancy_of_current_state += occupancy_matrix[current_state][current_action];
            const SType &state_obj = rmdp.get_state(current_state);
            for ( int current_action = 0; current_action < action_count; current_action++ ){
                if ( state_obj.is_valid(current_action) ){
                    const auto &action_obj = state_obj.get_action(current_action);
                    size_t num_outcomes = action_obj.outcome_count();
                    for ( int current_outcome = 0; current_outcome < num_outcomes; current_outcome++ ) {
                        const Transition &outcome_obj = action_obj.get_outcome(current_outcome);
                        prec_t current_outcome_weight = action_obj.get_weight(current_outcome);
                        for ( int current_transition = 0; current_transition < outcome_obj.size(); current_transition++)
                        {
                            const long target_state = outcome_obj.get_index(current_transition);
                            const prob_list_t &policy_target_state = policy[target_state];
                            const double transition_weight = outcome_obj.get_probability(current_transition);
                            (*new_occupancy_additions)[target_state][current_action] += occupancy_of_current_state *
                                                                              policy_target_state[current_action] *
                                                                              current_outcome_weight *
                                                                              transition_weight *
                                                                              pow( discount, t );
                        }
                    }
                }
            }
        }

        for ( int current_state = 0; current_state < state_count; current_state++ ) {
            for ( int current_action = 0; current_action < action_count; current_action++ )
                occupancy_matrix[current_state][current_action] += (*new_occupancy_additions)[current_state][current_action];
        }

        delete new_occupancy_additions;
    }

    return occupancy_matrix;
}

}}

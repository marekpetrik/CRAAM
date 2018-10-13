#pragma once

#include "craam/RMDP.hpp"

#include <eigen3/Eigen/Dense>
#include <rm/range.hpp>

namespace craam { namespace algorithms{

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
    /// \param state
    /// \param index
    /// \param policies
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
\param policies The policy (indvec) or a pair of the policy and the policy
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

} }

#pragma once

#include "../RMDP.hpp"

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "../cpp11-range-master/range.hpp"

namespace craam{namespace algorithms{

using namespace std;
using namespace boost::numeric;

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
        static_assert(!SType::requires_nature, "The type of state requires that the policy of nature is also provided.");
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
        static_assert(!SType::requires_nature, "The type of state requires that the policy of nature is also provided.");
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
inline ublas::matrix<prec_t>
transition_mat(const GRMDP<SType>& rmdp, const Policies& policies, bool transpose = false) {
    const size_t n = rmdp.state_count();
    ublas::matrix<prec_t> result = ublas::zero_matrix<prec_t>(n,n);

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
    auto&& initial_svec = init.probabilities_vector(n);
    ublas::vector<prec_t> initial_vec(n);

    // TODO: this is a wasteful copy operation
    copy(initial_svec.begin(), initial_svec.end(), initial_vec.data().begin());

    // get transition matrix
    ublas::matrix<prec_t> t_mat = transition_mat(rmdp, policies,true);

    // construct main matrix
    t_mat *= -discount;
    t_mat += ublas::identity_matrix<prec_t>(n);

    // solve set of linear equations
    ublas::permutation_matrix<prec_t> P(n);
    ublas::lu_factorize(t_mat,P);
    ublas::lu_substitute(t_mat,P,initial_vec);

    // copy the solution back to a vector
    copy(initial_vec.begin(), initial_vec.end(), initial_svec.begin());

    return initial_svec;
}

}}

#pragma once
#include <vector>
#include <istream>
#include <fstream>
#include <memory>
#include <tuple>
#include <armadillo>

#include "definitions.hpp"
#include "State.hpp"

using namespace std;

namespace craam {

/**
 Describes the behavior of nature in the uncertain MDP. Robust corresponds to the
 worst-case behavior of nature, optimistic corresponds the best case, and average
 represents a weighted mean of the returns.
 */
enum SolutionType {
    Robust = 0,
    Optimistic = 1,
    Average = 2
};

/**
    Represents a solution to a robust MDP.
 */
class Solution {
public:
    //TODO: rename outcomes and outcomes_dist to natpolicy (= nature policy)
    numvec valuefunction;
    indvec policy;                        // index of the actions for each states
    indvec outcomes;                      // index of the outcome for each state
    vector<numvec> outcome_dists;       // distribution of outcomes for each state
    prec_t residual;
    long iterations;
   
    Solution():
        valuefunction(0), policy(0), outcomes(0),
        outcome_dists(0),residual(-1),iterations(-1) {};

    Solution(numvec const& valuefunction, indvec const& policy,
             indvec const& outcomes, prec_t residual = -1, long iterations = -1) :
        valuefunction(valuefunction), policy(policy), outcomes(outcomes),
        outcome_dists(0),residual(residual),iterations(iterations) {};

    Solution(numvec const& valuefunction, indvec const& policy,
             const vector<numvec>& outcome_dists, prec_t residual = -1, long iterations = -1):
        valuefunction(valuefunction), policy(policy), outcomes(0),
        outcome_dists(outcome_dists),residual(residual),iterations(iterations){};

    prec_t total_return(const Transition& initial) const;
};

/**
    A robust Markov decision process. Contains methods for constructing and solving RMDPs.

    Some general assumptions:
    - Transition probabilities must be non-negative but do not need to add up to a specific value
    - Transitions with 0 probabilities may be omitted, except there must be at least one target state in each transition
    - State with no actions: A terminal state with value 0
    - Action with no outcomes: Terminates with an error
    - Outcome with no target states: Terminates with an error
 */
class RMDP{

public:

    RMDP(long state_count){
        /**
          Constructs the RMDP with a pre-allocated number of states. The state ids
          must be sequential and are constructed as needed.

          \param state_count The number of states.
         */

        states = vector<State>(state_count);
    };

    RMDP() : RMDP(0) {
        /**
          Constructs an empty RMDP.
         */
    };

    // adding transitions
    void add_transition(long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);
    void add_transition_d(long fromid, long actionid, long toid, prec_t probability, prec_t reward);
    void add_transitions(indvec const& fromids, indvec const& actionids, indvec const& outcomeids, indvec const& toids, numvec const& probs, numvec const& rews);

    // manipulate MDP attributes
    void set_distribution(long fromid, long actionid, numvec const& distribution, prec_t threshold);
    void set_threshold(long stateid, long actionid, prec_t threshold);
    void set_uniform_distribution(prec_t threshold);
    void set_uniform_thresholds(prec_t threshold);
    void set_reward(long stateid, long actionid, long outcomeid, long sampleid, prec_t reward);

    // querying parameters
    prec_t get_reward(long stateid, long actionid, long outcomeid, long sampleid) const;
    prec_t get_toid(long stateid, long actionid, long outcomeid, long sampleid) const;
    prec_t get_probability(long stateid, long actionid, long outcomeid, long sampleid) const;
    Transition& get_transition(long stateid, long actionid, long outcomeid);
    const Transition& get_transition(long stateid, long actionid, long outcomeid) const;
    prec_t get_threshold(long stateid, long actionid) const;

    // object counts
    size_t state_count() const;
    size_t action_count(long stateid) const;
    size_t outcome_count(long stateid, long actionid) const;
    size_t transition_count(long stateid, long actionid, long outcomeid) const;

    // normalization
    bool is_normalized() const;
    void normalize();

    // writing a reading files
    static unique_ptr<RMDP> transitions_from_csv(istream& input, bool header = true);
    void transitions_to_csv(ostream& output, bool header = true) const;
    void transitions_to_csv_file(const string& filename, bool header = true) const;

    // copying
    // TODO: deprecate these methods and replace with the copy constructor
    unique_ptr<RMDP> copy() const;
    void copy_into(RMDP& result) const;

    // string representation
    string to_string() const;

    // fixed-policy functions
    numvec ofreq_mat(const Transition& init, prec_t discount, const indvec& policy, const indvec& nature) const;
    numvec rewards_state(const indvec& policy, const indvec& nature) const;
    unique_ptr<arma::SpMat<prec_t>> transition_mat(const indvec& policy, const indvec& nature) const;
    unique_ptr<arma::SpMat<prec_t>> transition_mat_t(const indvec& policy, const indvec& nature) const;

    // value iteration
    Solution vi_gs_rob(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_gs_opt(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_gs_ave(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_gs_l1_rob(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_gs_l1_opt(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    Solution vi_jac_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_jac_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_jac_ave(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_jac_l1_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    Solution vi_jac_l1_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;


    // modified policy iteration
    Solution mpi_jac_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
    Solution mpi_jac_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
    Solution mpi_jac_ave(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
    Solution mpi_jac_l1_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
    Solution mpi_jac_l1_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;

protected:
    vector<State> states;

    template<SolutionType type> 
    Solution vi_gs_gen(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    template<SolutionType type, NatureConstr nature> 
    Solution vi_gs_cst(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    template<SolutionType type, NatureConstr nature> 
    Solution mpi_jac_cst(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
    template<SolutionType type,NatureConstr nature> 
    Solution vi_jac_cst(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const; 
    template<SolutionType type> 
    Solution vi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
    template<SolutionType type> 
    Solution mpi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;

};
}

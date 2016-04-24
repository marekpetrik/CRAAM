#pragma once
#include "definitions.hpp"

#include <vector>
#include <istream>
#include <fstream>
#include <memory>
#include <tuple>
#include <cassert>

#include "State.hpp"
#include <boost/numeric/ublas/matrix.hpp>


using namespace std;
using namespace boost::numeric;

namespace craam {


/**
 Describes the behavior of nature in the uncertain MDP. Robust corresponds to the
 worst-case behavior of nature, optimistic corresponds the best case, and average
 represents a weighted mean of the returns.
 */
enum class Uncertainty {
    Robust = 0,
    Optimistic = 1,
    Average = 2
};

/** A solution to a robust MDP.  */
class Solution {
public:
    numvec valuefunction;
    indvec policy;                        // index of the actions for each states
    indvec outcomes;                      // index of the outcome for each state
    vector<numvec> natpolicy;       // distribution of outcomes for each state
    prec_t residual;
    long iterations;

    Solution():
        valuefunction(0), policy(0), outcomes(0),
        natpolicy(0),residual(-1),iterations(-1) {};

    Solution(numvec const& valuefunction, indvec const& policy,
             indvec const& outcomes, prec_t residual = -1, long iterations = -1) :
        valuefunction(valuefunction), policy(policy), outcomes(outcomes),
        natpolicy(0),residual(residual),iterations(iterations) {};

    Solution(numvec const& valuefunction, indvec const& policy,
             const vector<numvec>& natpolicy, prec_t residual = -1, long iterations = -1):
        valuefunction(valuefunction), policy(policy), outcomes(0),
        natpolicy(natpolicy),residual(residual),iterations(iterations){};

    /**
    Computes the total return of the solution given the initial
    distribution.

    \param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const;
};

/**
A robust Markov decision process. Contains methods for constructing and solving RMDPs.

Some general assumptions:
    - Transition probabilities must be non-negative but do not need to add
        up to a specific value
    - Transitions with 0 probabilities may be omitted, except there must
        be at least one target state in each transition
    - State with no actions: A terminal state with value 0
    - Action with no outcomes: Terminates with an error for uncertain models, but
                               assumes 0 return for regular models.
    - Outcome with no target states: Terminates with an error
 */
class RMDP{
public:
    /**
    Constructs the RMDP with a pre-allocated number of states.
    The state ids must be sequential and are constructed as needed.
    \param state_count The number of states.
    */
    RMDP(long state_count){
        states = vector<State>(state_count);
    };

    /** Constructs an empty RMDP. */
    RMDP() : RMDP(0) {};

    // adding transitions
    /**
    Adds a transition probability
    \param fromid Starting state ID
    \param actionid Action ID
    \param outcomeid Outcome ID (A single outcome corresponds to a regular MDP)
    \param toid Destination ID
    \param probability Probability of the transition (must be non-negative)
    \param reward The reward associated with the transition.
     */
    void add_transition(long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);

    /** Adds a non-robust transition.  */
    void add_transition_d(long fromid, long actionid, long toid, prec_t probability, prec_t reward);

    /**
    Add multiple samples (transitions) to the MDP definition
    \param fromids Starting state ids
    \param outcomeis IDs used of the outcomes
    \param toids Destination state ids
    \param actionids
    \param probs Probabilities of the transitions
    \param rews Rewards of the transitions
     */
    void add_transitions(indvec const& fromids, indvec const& actionids, indvec const& outcomeids,
                         indvec const& toids, numvec const& probs, numvec const& rews);

    /** Assures that the MDP state exists and if it does not, then it is created */
    void assure_state_exists(long stateid);

    // manipulate weights
    /**
    Sets the distribution for outcomes for each state and
    action to be uniform. It also sets the threshold to be the same
    for all states. */
    void set_uniform_distribution(prec_t threshold);
    /** Sets thresholds for all states uniformly */
    void set_uniform_thresholds(prec_t threshold);

    // get parameters
    /** Returns the transition. The transition must exist. */
    Transition& get_transition(long stateid, long actionid, long outcomeid);
    /** Returns the transition. The transition must exist. */
    const Transition& get_transition(long stateid, long actionid, long outcomeid) const;

    /** Return a transition for state, action, and outcome. It is created if necessary. */
    Transition& create_transition(long stateid, long actionid, long outcomeid);
    State& get_state(long stateid) {assert(stateid >= 0 && stateid < (long) states.size()); return states[stateid];}
    const State& get_state(long stateid) const {assert(stateid >= 0 && stateid < (long) states.size()); return states[stateid];};

    // object counts
    size_t state_count() const;

    // normalization of transition probabilities
    /**
    Check if all transitions in the process sum to one.
    Note that if there are no actions, or no outcomes for a state,
    the RMDP still may be normalized.
    \return True if and only if all transitions are normalized.
     */
    bool is_normalized() const;

    /** Normalize all transitions to sum to one for all states, actions, outcomes. */
    void normalize();

    // writing a reading files
    /**
    Loads an RMDP definition from a simple csv file.States, actions, and
    outcomes are identified by 0-based ids. The columns are separated by
    commas, and rows by new lines.

    The file is formatted with the following columns:
    idstatefrom, idaction, idoutcome, idstateto, probability, reward

    Note that outcome distributions are not restored.

    \param input Source of the RMDP
    \param header Whether the first line of the file represents the header.
                    The column names are not checked for correctness or number!
     */
    static unique_ptr<RMDP> from_csv(istream& input, bool header = true);

    /**
    Loads the transition probabilities and rewards from a CSV file
    \param filename Name of the file
    \param header Whether to create a header of the file too
     */
    static unique_ptr<RMDP> from_csv_file(const string& filename, bool header = true);

    /**
    Saves the model to a stream as a simple csv file. States, actions, and outcomes
    are identified by 0-based ids.

    The columns are separated by commas, and rows by new lines.

    The file is formatted with the following columns:
    idstatefrom, idaction, idoutcome, idstateto, probability, reward

    Exported and imported MDP will be be slightly different. Since action/transitions
    will not be exported if there are no actions for the state. However, when
    there is data for action 1 and action 3, action 2 will be created with no outcomes.

    Note that outcome distributions are not saved.

    \param output Output for the stream
    \param header Whether the header should be written as the
          first line of the file represents the header.
     */
    void to_csv(ostream& output, bool header = true) const;
   /**
    Saves the transition probabilities and rewards to a CSV file

    \param filename Name of the file
    \param header Whether to create a header of the file too
     */
    void to_csv_file(const string& filename, bool header = true) const;

    // string representation
    /**
    Returns a brief string representation of the MDP.
    This method is mostly suitable for analyzing small MDPs.
    */
    string to_string() const;

    // fixed-policy functions
    /**
    Computes occupancy frequencies using matrix representation of transition
    probabilities.

    This method does not scale to larger state spaces

    \param init Initial distribution (alpha)
    \param discount Discount factor (gamma)
    \param policy Policy of the decision maker
    \param nature Policy of nature
    */
    numvec ofreq_mat(const Transition& init, prec_t discount, const indvec& policy, const indvec& nature) const;

    /**
    Constructs the rewards vector for each state for the RMDP.

    \param policy Policy of the decision maker
    \param nature Policy of nature
     */
    numvec rewards_state(const indvec& policy, const indvec& nature) const;

    /**
    Checks if the policy and nature's policy are both correct. If
    not, the function returns the first state with an incorrect
    action and outcome. Otherwise the function return -1.

    Action and outcome can be arbitrary for terminal states.
    */
    long assert_policy_correct(indvec policy, indvec natpolicy) const;

    /**
    Constructs the transition matrix for the policy.

    \param policy Policy of the decision maker
    \param nature Policy of the nature
    */
    unique_ptr<ublas::matrix<prec_t>> transition_mat(const indvec& policy, const indvec& nature) const;

    /**
    Constructs a transpose of the transition matrix for the policy.

    \param policy Policy of the decision maker
    \param nature Policy of the nature
    */
    unique_ptr<ublas::matrix<prec_t>> transition_mat_t(const indvec& policy, const indvec& nature) const;

    /**
    Value function evaluation using Jacobi iteration for a fixed policy.
    and nature.

    \param valuefunction Initial value function
    \param discount Discount factor
    \param policy Decision-maker's policy
    \param natpolicy Nature's policy
    \param iterations Maximal number of inner loop value iterations
    \param maxresidual Stop the inner policy iteration when
            the residual drops below this threshold.
    \return Computed (approximate) solution (value function)
     */
    Solution vi_jac_fix(const numvec& valuefunction, prec_t discount, const indvec& policy,
                        const indvec& natpolicy, unsigned long iterations=MAXITER,
                        prec_t maxresidual=SOLPREC) const;

    /**
    Value function evaluation using Jacobi iteration for a fixed policy.
    and average value for the nature.

    \param valuefunction Initial value function
    \param discount Discount factor
    \param policy Decision-maker's policy
    \param natpolicy Nature's policy
    \param iterations Maximal number of inner loop value iterations
    \param maxresidual Stop the inner policy iteration when
            the residual drops below this threshold.
    \return Computed (approximate) solution (value function)
     */
    Solution vi_jac_fix_ave(const numvec& valuefunction, prec_t discount, const indvec& policy,
                            unsigned long iterations=MAXITER,
                            prec_t maxresidual=SOLPREC) const;

    // value iteration - GS
    /**
    Gauss-Seidel value iteration variant (not parallelized). The outcomes are
    selected using worst-case nature.

    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to parallelize easily.

    \param valuefunction Initial value function. Passed by value, because it is modified.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_gs_rob(numvec valuefunction, prec_t discount,
                       unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Gauss-Seidel value iteration variant (not parallelized). The outcomes are
    selected using best-case nature.

    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to parallelize easily.

    \param valuefunction Initial value function. Passed by value, because it is modified.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_gs_opt(numvec valuefunction, prec_t discount,
                       unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Gauss-Seidel value iteration variant (not parallelized). The outcomes are
    selected using average-case nature.

    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to paralelize easily.

    \param valuefunction Initial value function. Passed by value, because it is modified.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_gs_ave(numvec valuefunction, prec_t discount,
                       unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Robust Gauss-Seidel value iteration variant (not parallelized). The natures policy is
    constrained using L1 constraints and is worst-case.

    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to parallelize.

    \param valuefunction Initial value function. Passed by value, because it is modified.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_gs_l1_rob(numvec valuefunction, prec_t discount,
                          unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Optimistic Gauss-Seidel value iteration variant (not parallelized). The natures policy is
    constrained using L1 constraints and is best-case.

    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to parallelize.

    This is a generic version, which works for best/worst-case optimization and
    arbitrary constraints on nature (given by the function nature). Average case constrained
    nature is not supported.

    \param valuefunction Initial value function. Passed by value, because it is modified.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_gs_l1_opt(numvec valuefunction, prec_t discount,
                          unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;

    // value iteration _ JAC
    /**
    Robust Jacobi value iteration variant. The nature behaves as worst-case.
    This method uses OpenMP to parallelize the computation.

    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_jac_rob(numvec const& valuefunction, prec_t discount,
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Optimistic Jacobi value iteration variant. The nature behaves as best-case.
    This method uses OpenMP to parallelize the computation.

    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_jac_opt(numvec const& valuefunction, prec_t discount,
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Average Jacobi value iteration variant. The nature behaves as average-case.
    This method uses OpenMP to parallelize the computation.

    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_jac_ave(numvec const& valuefunction, prec_t discount,
                        unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Robust Jacobi value iteration variant with constrained nature. The nature is constrained
    by an L1 norm.

    This method uses OpenMP to parallelize the computation.

    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    Solution vi_jac_l1_rob(numvec const& valuefunction, prec_t discount,
                           unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;
    /**
    Optimistic Jacobi value iteration variant with constrained nature.
    The nature is constrained by an L1 norm.

    This method uses OpenMP to parallelize the computation.

    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
    */
    Solution vi_jac_l1_opt(numvec const& valuefunction, prec_t discount,
                           unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;

    // modified policy iteration
    /**
    Robust modified policy iteration using Jacobi value iteration in the inner loop.
    The nature behaves as worst-case.

    This method generalizes modified policy iteration to robust MDPs.
    In the value iteration step, both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    Solution mpi_jac_rob(numvec const& valuefunction, prec_t discount,
                         unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                         unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2) const;

    /**
    Optimistic modified policy iteration using Jacobi value iteration in the inner loop.
    The nature behaves as best-case.

    This method generalizes modified policy iteration to robust MDPs.
    In the value iteration step, both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    Solution mpi_jac_opt(numvec const& valuefunction, prec_t discount,
                         unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                         unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2) const;
    /**
    Average modified policy iteration using Jacobi value iteration in the inner loop.
    The nature behaves as average-case.

    This method generalizes modified policy iteration to robust MDPs.
    In the value iteration step, both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    Solution mpi_jac_ave(numvec const& valuefunction, prec_t discount,
                         unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                         unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2) const;
    /**
    Robust modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
    The constraints are defined by the L1 norm and the nature is worst-case.

    This method generalized modified policy iteration to the robust MDP. In the value iteration step,
    both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    Solution mpi_jac_l1_rob(numvec const& valuefunction, prec_t discount,
                            unsigned long iterations_pi=MAXITER, prec_t maxresidual_pi=SOLPREC,
                            unsigned long iterations_vi=MAXITER, prec_t maxresidual_vi=SOLPREC/2) const;
    /**
    Optimistic modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
    The constraints are defined by the L1 norm and the nature is best-case.

    This method generalized modified policy iteration to the robust MDP. In the value iteration step,
    both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
    */
    Solution mpi_jac_l1_opt(numvec const& valuefunction,
                            prec_t discount,
                            unsigned long iterations_pi=MAXITER,
                            prec_t maxresidual_pi=SOLPREC,
                            unsigned long iterations_vi=MAXITER,
                            prec_t maxresidual_vi=SOLPREC/2) const;
public:
    vector<State> states;
protected:

    /**
    Gauss-Seidel value iteration variant (not parallelized). This is a generic function,
    which can compute any solution type (robust, optimistic, or average).

    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to parallelize.

    \param valuefunction Initial value function. Passed by value,
                        because it is modified. If it has size 0, then it is assumed
                        to be all 0s.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
    \return Computed (approximate) solution
     */
    template<Uncertainty type>
    Solution vi_gs_gen(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    /**
    Gauss-Seidel value iteration variant with constrained nature(not parallelized).
    The natures policy is constrained, given by the function nature.

    Because this function updates the array value during the iteration, it may be
    difficult to parallelize.

    This is a generic version, which works for best/worst-case optimization and
    arbitrary constraints on nature (given by the function nature). Average case constrained
    nature is not supported.

    \param valuefunction Initial value function. Passed by value, because it is modified. When
                           it has zero length, it is assumed to be all zeros.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
    \return Computed (approximate) solution
     */
    template<Uncertainty type, NatureConstr nature>
    Solution vi_gs_cst(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    /**
    Modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
    The template determines the constraints (by the parameter nature) and the type
    of nature (by the parameter type)

    This method generalized modified policy iteration to the robust MDP. In the value iteration step,
    both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    template<Uncertainty type, NatureConstr nature>
    Solution mpi_jac_cst(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                         unsigned long iterations_vi, prec_t maxresidual_vi) const;

    /**
    Jacobi value iteration variant with constrained nature. The outcomes are
    selected using nature function.

    This method uses OpenMP to parallelize the computation.

    This is a generic version, which works for best/worst-case optimization and
    arbitrary constraints on nature (given by the function nature). Average case constrained
    nature is not supported.

    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
    \return Computed (approximate) solution
     */
    template<Uncertainty type,NatureConstr nature>
    Solution vi_jac_cst(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    /**
    Jacobi value iteration variant. The behavior of the nature depends on the values
    of parameter type. This method uses OpenMP to parallelize the computation.

    \param valuefunction Initial value function, if size zero, then considered to be all zeros.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
    \return Computed (approximate) solution
     */
    template<Uncertainty type>
    Solution vi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    /**
    Modified policy iteration using Jacobi value iteration in the inner loop.
    The template parameter type determines the behavior of nature.

    This method generalizes modified policy iteration to robust MDPs.
    In the value iteration step, both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

    \param valuefunction Initial value function, use a vector of length 0 if the value is not provided
    \param discount Discount factor
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    template<Uncertainty type>
    Solution mpi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                         unsigned long iterations_vi, prec_t maxresidual_vi) const;

};

// **************************************************************************************
//  Generic MDP Class
// **************************************************************************************

/** A solution to a robust MDP.  */
template<typename ActionId, typename OutcomeId>
class GSolution {
public:
    numvec valuefunction;
    vector<ActionId> policy;                        // index of the actions for each states
    vector<OutcomeId> outcomes;                      // index of the outcome for each state
    prec_t residual;
    long iterations;

    GSolution():
        valuefunction(0), policy(0), outcomes(0),
        residual(-1),iterations(-1) {};

    GSolution(numvec const& valuefunction, const vector<ActionId>& policy,
             const vector<OutcomeId>& outcomes, prec_t residual = -1, long iterations = -1) :
        valuefunction(valuefunction), policy(policy), outcomes(outcomes),
        residual(residual),iterations(iterations) {};

    /**
    Computes the total return of the solution given the initial
    distribution.
    \param initial The initial distribution
     */
    prec_t total_return(const Transition& initial) const{
        if(initial.max_index() >= (long) valuefunction.size())
            throw invalid_argument("Too many indexes in the initial distribution.");
        return initial.compute_value(valuefunction);
    };
};

/**
A general robust Markov decision process. Contains methods for constructing and solving RMDPs.

Some general assumptions (may depend on the state and action classes):
    - Transition probabilities must be non-negative but do not need to add
        up to a specific value
    - Transitions with 0 probabilities may be omitted, except there must
        be at least one target state in each transition
    - State with no actions: A terminal state with value 0
    - Action with no outcomes: Terminates with an error for uncertain models, but
                               assumes 0 return for regular models.
    - Outcome with no target states: Terminates with an error

\tparam SType Type of state, determines s-rectangularity or s,a-rectangularity and
        also the type of the outcome and action constraints
 */
template<class SType>
class GRMDP{
protected:
    /** Internal list of states */
    vector<SType> states;

public:
    /** Type defining action to take in which state */
    typedef vector<typename SType::ActionId> ActionPolicy;
    /** Type defining outcome to take in which state*/
    typedef vector<typename SType::OutcomeId> OutcomePolicy;
    /** Type of solution */
    typedef GSolution<typename SType::ActionId, typename SType::OutcomeId> SolType;

    /**
    Constructs the RMDP with a pre-allocated number of states. All
    states are initially terminal.
    \param state_count The number of states.
    */
    GRMDP(long state_count) : states(state_count){};

    /** Constructs an empty RMDP. */
    GRMDP(){};

    /**
    Assures that the MDP state exists and if it does not, then it is created.
    States with intermediate ids are also created
    \return The new state
    */
    SType create_state(long stateid);

    /**
    Creates a new state at the end of the states
    \return The new state
    */
    SType create_state(){ return create_state(states.size());};

    /** Number of states */
    size_t state_count() const {return states.size();};

    /** Retrieves an existing state */
    const SType& get_state(long stateid) const {assert(stateid >= 0 && size_t(stateid) < state_count());
                                                return states[stateid];};

    /** Retrieves an existing state */
    SType& get_state(long stateid) {assert(stateid >= 0 && size_t(stateid) < state_count());
    return states[stateid];};

    /** Returns list of all states */
    const vector<SType> get_states() const {return states;};

    /**
    Check if all transitions in the process sum to one.
    Note that if there are no actions, or no outcomes for a state,
    the RMDP still may be normalized.
    \return True if and only if all transitions are normalized.
     */
    bool is_normalized() const;

    /** Normalize all transitions to sum to one for all states, actions, outcomes. */
    void normalize();

    /**
    Computes occupancy frequencies using matrix representation of transition
    probabilities. This method does not scale to larger state spaces
    \param init Initial distribution (alpha)
    \param discount Discount factor (gamma)
    \param policy Policy of the decision maker
    \param nature Policy of nature
    */
    numvec ofreq_mat(const Transition& init, prec_t discount,
                     const ActionPolicy& policy, const OutcomePolicy& nature) const;

    /**
    Constructs the rewards vector for each state for the RMDP.
    \param policy Policy of the decision maker
    \param nature Policy of nature
     */
    numvec rewards_state(const ActionPolicy& policy, const OutcomePolicy& nature) const;

    /**
    Checks if the policy and nature's policy are both correct.
    Action and outcome can be arbitrary for terminal states.
    \return If incorrect, the function returns the first state with an incorrect
            action and outcome. Otherwise the function return -1.
    */
    long is_policy_correct(const ActionPolicy& policy,
                           const OutcomePolicy& natpolicy) const;

    // ----------------------------------------------
    // Solution methods
    // ----------------------------------------------

    /**
    Gauss-Seidel value iteration variant (not parallelized).
    This function is suitable for computing the value function of a finite state MDP. If
    the states are ordered correctly, one iteration is enough to compute the optimal value function.
    Since the value function is updated from the first state to the last, the states should be ordered
    in reverse temporal order.

    Because this function updates the array value during the iteration, it may be
    difficult to paralelize easily.
    \param uncert Type of realization of the uncertainty
    \param discount Discount factor.
    \param valuefunction Initial value function. Passed by value, because it is modified.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    SolType vi_gs(Uncertainty uncert,
                  prec_t discount,
                  numvec valuefunction=numvec(0),
                    unsigned long iterations=MAXITER, prec_t maxresidual=SOLPREC) const;

    /**
    Jacobi value iteration variant. This method uses OpenMP to parallelize the computation.
    \param uncert Type of realization of the uncertainty
    \param valuefunction Initial value function.
    \param discount Discount factor.
    \param iterations Maximal number of iterations to run
    \param maxresidual Stop when the maximal residual falls below this value.
     */
    SolType vi_jac(Uncertainty uncert,
                   prec_t discount,
                   const numvec& valuefunction=numvec(0),
                    unsigned long iterations=MAXITER,
                    prec_t maxresidual=SOLPREC) const;

    /**
    Modified policy iteration using Jacobi value iteration in the inner loop.
    This method generalizes modified policy iteration to robust MDPs.
    In the value iteration step, both the action *and* the outcome are fixed.

    Note that the total number of iterations will be bounded by iterations_pi * iterations_vi
    \param uncert Type of realization of the uncertainty
    \param discount Discount factor
    \param valuefunction Initial value function
    \param iterations_pi Maximal number of policy iteration steps
    \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
    \param iterations_vi Maximal number of inner loop value iterations
    \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                This value should be smaller than maxresidual_pi
    \return Computed (approximate) solution
     */
    SolType mpi_jac(Uncertainty uncert,
                    prec_t discount,
                    const numvec& valuefunction=numvec(0),
                    unsigned long iterations_pi=MAXITER,
                    prec_t maxresidual_pi=SOLPREC,
                    unsigned long iterations_vi=MAXITER,
                    prec_t maxresidual_vi=SOLPREC/2) const;

    /**
    Value function evaluation using Jacobi iteration for a fixed policy.
    and nature.
    \param valuefunction Initial value function
    \param discount Discount factor
    \param policy Decision-maker's policy
    \param natpolicy Nature's policy
    \param iterations Maximal number of inner loop value iterations
    \param maxresidual Stop the inner policy iteration when
            the residual drops below this threshold.
    \return Computed (approximate) solution (value function)
     */
    SolType vi_jac_fix(prec_t discount,
                        const ActionPolicy& policy,
                        const OutcomePolicy& natpolicy,
                        const numvec& valuefunction=numvec(0),
                        unsigned long iterations=MAXITER,
                        prec_t maxresidual=SOLPREC) const;

    /**
    Value function evaluation using Jacobi iteration for a fixed policy
    and uncertainty realization type.
    \param uncert Type of realization of the uncertainty
    \param valuefunction Initial value function
    \param discount Discount factor
    \param policy Decision-maker's policy
    \param iterations Maximal number of inner loop value iterations
    \param maxresidual Stop the inner policy iteration when
            the residual drops below this threshold.
    \return Computed (approximate) solution (value function)
     */
    SolType vi_jac_fix(Uncertainty uncert,
                       prec_t discount,
                       const ActionPolicy& policy,
                       const numvec& valuefunction=numvec(0),
                       unsigned long iterations=MAXITER,
                       prec_t maxresidual=SOLPREC) const;

    /**
    Constructs the transition matrix for the policy.
    \param policy Policy of the decision maker
    \param nature Policy of the nature
    */
    unique_ptr<ublas::matrix<prec_t>>
        transition_mat(const ActionPolicy& policy,
                       const OutcomePolicy& nature) const;

    /**
    Constructs a transpose of the transition matrix for the policy.
    \param policy Policy of the decision maker
    \param nature Policy of the nature
    */
    unique_ptr<ublas::matrix<prec_t>>
        transition_mat_t(const ActionPolicy& policy,
                         const OutcomePolicy& nature) const;

    // ----------------------------------------------
    // Reading and writing files
    // ----------------------------------------------
    /**
    Loads an RMDP definition from a simple csv file.States, actions, and
    outcomes are identified by 0-based ids. The columns are separated by
    commas, and rows by new lines.

    The file is formatted with the following columns:
    idstatefrom, idaction, idoutcome, idstateto, probability, reward

    Note that outcome distributions are not restored.
    \param result Where to store the loaded MDP
    \param input Source of the RMDP
    \param header Whether the first line of the file represents the header.
                    The column names are not checked for correctness or number!
     */
    static void from_csv(GRMDP<SType>& result, istream& input, bool header = true);

    /**
    Loads the transition probabilities and rewards from a CSV file
    \param filename Name of the file
    \param header Whether to create a header of the file too
     */
    static void from_csv_file(GRMDP<SType>& result, const string& filename, bool header = true);

    /**
    Saves the model to a stream as a simple csv file. States, actions, and outcomes
    are identified by 0-based ids. Columns are separated by commas, and rows by new lines.

    The file is formatted with the following columns:
    idstatefrom, idaction, idoutcome, idstateto, probability, reward

    Exported and imported MDP will be be slightly different. Since action/transitions
    will not be exported if there are no actions for the state. However, when
    there is data for action 1 and action 3, action 2 will be created with no outcomes.

    Note that outcome distributions are not saved.

    \param output Output for the stream
    \param header Whether the header should be written as the
          first line of the file represents the header.
    */
    void to_csv(ostream& output, bool header = true) const;

    /**
    Saves the transition probabilities and rewards to a CSV file
    \param filename Name of the file
    \param header Whether to create a header of the file too
     */
    void to_csv_file(const string& filename, bool header = true) const;

    // string representation
    /**
    Returns a brief string representation of the MDP.
    This method is mostly suitable for analyzing small MDPs.
    */
    string to_string() const;
};


/**
Adds a transition probability.
\param gmdp MDP to add transition to
\param fromid Starting state ID
\param actionid Action ID
\param outcomeid Outcome ID (A single outcome corresponds to a regular MDP)
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
 */
template<SType>
void add_transition(GRMDP& grmdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){
{

    auto s = grmdp.create_state(fromid);
    grmdp.create_state(toid);

    .add_action(actionid, outcomeid, toid, probability, reward);
}


}

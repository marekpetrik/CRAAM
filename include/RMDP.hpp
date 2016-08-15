#pragma once

#include "State.hpp"

#include <vector>
#include <istream>
#include <fstream>
#include <memory>
#include <tuple>
#include <cassert>

#include <boost/numeric/ublas/matrix.hpp>

/// Main namespace which includes modeling a solving functionality
namespace craam {

using namespace std;
using namespace boost::numeric;

/**
Describes the behavior of nature in the uncertain MDP. Robust corresponds to the
worst-case behavior of nature, optimistic corresponds the best case, and average
represents a weighted mean of the returns.
*/
enum class Uncertainty {
    /// Treat uncertainty as a worst case
    Robust = 0,
    /// Treat uncertainty as a best case
    Optimistic = 1,
    /// Average over uncertain outcomes (type of average depends on the type)
    Average = 2
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

    Computes it based on the value function.

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
    - Invalid actions are ignored
    - Behavior for a state with all invalid actions is not defined

\tparam SType Type of state, determines s-rectangularity or s,a-rectangularity and
        also the type of the outcome and action constraints
 */
template<class SType>
class GRMDP{
protected:
    /** Internal list of states */
    vector<SType> states;

public:

    /** Action identifier in a policy. Copies type from state type. */
    typedef typename SType::ActionId ActionId;
    /** Action identifier in a policy. Copies type from state type. */
    typedef typename SType::OutcomeId OutcomeId;

    /** Decision-maker's policy: Which action to take in which state.  */
    typedef vector<ActionId> ActionPolicy;
    /** Nature's policy: Which outcome to take in which state.  */
    typedef vector<OutcomeId> OutcomePolicy;
    /** Solution type */
    typedef GSolution<typename SType::ActionId, typename SType::OutcomeId>
                SolType;

    /**
    Constructs the RMDP with a pre-allocated number of states. All
    states are initially terminal.
    \param state_count The initial number of states, which dynamically
                        increases as more transitions are added. All initial
                        states are terminal.
    */
    GRMDP(long state_count) : states(state_count){};

    /** Constructs an empty RMDP. */
    GRMDP(){};

    /**
    Assures that the MDP state exists and if it does not, then it is created.
    States with intermediate ids are also created
    \return The new state
    */
    SType& create_state(long stateid);

    /**
    Creates a new state at the end of the states
    \return The new state
    */
    SType& create_state(){ return create_state(states.size());};

    /** Number of states */
    size_t state_count() const {return states.size();};

    /** Number of states */
    size_t size() const {return state_count();};

    /** Retrieves an existing state */
    const SType& get_state(long stateid) const {assert(stateid >= 0 && size_t(stateid) < state_count());
                                                return states[stateid];};

    /** Retrieves an existing state */
    const SType& operator[](long stateid) const {return get_state(stateid);};


    /** Retrieves an existing state */
    SType& get_state(long stateid) {assert(stateid >= 0 && size_t(stateid) < state_count());
                                    return states[stateid];};

    /** Retrieves an existing state */
    SType& operator[](long stateid){return get_state(stateid);};

    /** \returns list of all states */
    const vector<SType>& get_states() const {return states;};

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
    Gauss-Seidel varaint of value iteration (not parallelized).
    
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
                  unsigned long iterations=MAXITER,
                  prec_t maxresidual=SOLPREC) const;

    /**
    Jacobi variant of value iteration. This method uses OpenMP to parallelize the computation.
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
    \param show_progress Whether to report on progress during the computation
    \return Computed (approximate) solution
     */
    SolType mpi_jac(Uncertainty uncert,
                    prec_t discount,
                    const numvec& valuefunction=numvec(0),
                    unsigned long iterations_pi=MAXITER,
                    prec_t maxresidual_pi=SOLPREC,
                    unsigned long iterations_vi=MAXITER,
                    prec_t maxresidual_vi=SOLPREC/2,
                    bool show_progress=false) const;

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

    // TODO: a function like this could be useful
    /*
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
    //SolType vi_jac_fix(Uncertainty uncert,
    //                   prec_t discount,
    //                   const ActionPolicy& policy,
    //                   const numvec& valuefunction=numvec(0),
    //                   unsigned long iterations=MAXITER,
    //                   prec_t maxresidual=SOLPREC) const;

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
    Returns a brief string representation of the RMDP.
    This method is mostly suitable for analyzing small RMDPs.
    */
    string to_string() const;

    /**
    Returns a json representation of the RMDP.
    This method is mostly suitable to analyzing small RMDPs.
    */
    string to_json() const;
};

// **********************************************************************
// *********************    TEMPLATE DECLARATIONS    ********************
// **********************************************************************

/**
Regular MDP with discrete actions and one outcome per action

    ActionId = long
    OutcomeId = long

    ActionPolicy = vector<ActionId>
    OutcomePolicy = vector<OutcomeId>

Uncertainty type is ignored in these methods.
*/
typedef GRMDP<RegularState> MDP;

/**
An uncertain MDP with discrete robustness. See craam::DiscreteRobustState
*/
typedef GRMDP<DiscreteRobustState> RMDP_D;

/**
An uncertain MDP with L1 constrained robustness. See craam::L1RobustState.
*/
typedef GRMDP<L1RobustState> RMDP_L1;

/// Solution with discrete action and outcome policies
typedef GSolution<long, long> SolutionDscDsc;
/// Solution with discrete action and randomized outcome policy
typedef GSolution<long, numvec> SolutionDscProb;


}

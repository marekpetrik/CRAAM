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
#include "cpp11-range-master/range.hpp"


namespace craam {

using namespace std;
using namespace boost::numeric;
using namespace util::lang;

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


/// **************************************************************************************
///  Generic MDP Class
/// **************************************************************************************

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

    /** Action identifier in a policy. Copies type from state type. */
    typedef typename SType::ActionId ActionId;
    /** Action identifier in a policy. Copies type from state type. */
    typedef typename SType::OutcomeId OutcomeId;

    /** Which action to take in which state. Decision-maker's policy */
    typedef vector<ActionId> ActionPolicy;
    /** Which outcome to take in which state. Nature's policy */
    typedef vector<OutcomeId> OutcomePolicy;
    /** Solution type */
    typedef GSolution<typename SType::ActionId, typename SType::OutcomeId>
                SolType;

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

    /** Returns list of all states */
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

    /// ----------------------------------------------
    /// Solution methods
    /// ----------------------------------------------

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
                  unsigned long iterations=MAXITER,
                  prec_t maxresidual=SOLPREC) const;

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

    /// ----------------------------------------------
    /// Reading and writing files
    /// ----------------------------------------------

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

/// **********************************************************************
/// *********************    TEMPLATE DECLARATIONS    ********************
/// **********************************************************************

/**
Regular MDP with discrete actions and one outcome per action

    ActionId = long
    OutcomeId = long

    ActionPolicy = vector<ActionId>
    OutcomePolicy = vector<OutcomeId>
*/
typedef GRMDP<RegularState> MDP;
typedef GRMDP<DiscreteRobustState> RMDP_D;
typedef GRMDP<L1RobustState> RMDP_L1;


/// **********************************************************************
/// ***********************    HELPER FUNCTIONS    ***********************
/// **********************************************************************

/**
Adds a transition probability for a model with no outcomes.
\param mdp model to add the transition to
\param fromid Starting state ID
\param actionid Action ID
\param outcomeid Outcome ID (A single outcome corresponds to a regular MDP)
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
*/
template<class Model>
void add_transition(Model& mdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){

    // make sure that the destination state exists
    mdp.create_state(toid);

    auto& state_from = mdp.create_state(fromid);
    auto& action = state_from.create_action(actionid);
    Transition& outcome = action.create_outcome(outcomeid);
    outcome.add_sample(toid,probability,reward);
}

/**
Adds a transition probability for a particular outcome.
\param mdp model to add the transition to
\param fromid Starting state ID
\param actionid Action ID
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
*/
template<class Model>
void add_transition(Model& mdp, long fromid, long actionid, long toid, prec_t probability, prec_t reward){
    add_transition<Model>(mdp, fromid, actionid, 0l, toid, probability, reward);
}

/// **********************************************************************
/// ***********************    HELPER FUNCTIONS    ***********************
/// **********************************************************************

/**
Loads an RMDP definition from a simple csv file.States, actions, and
outcomes are identified by 0-based ids. The columns are separated by
commas, and rows by new lines.

The file is formatted with the following columns:
idstatefrom, idaction, idoutcome, idstateto, probability, reward

Note that outcome distributions are not restored.
\param mdp Model output (also returned)
\param input Source of the RMDP
\param header Whether the first line of the file represents the header.
                The column names are not checked for correctness or number!
\returns The input model
 */
template<class Model>
Model& from_csv(Model& mdp, istream& input, bool header = true){
{
    string line;
    // skip the first row if so instructed
    if(header) input >> line;

    input >> line;
    while(input.good()){
        string cellstring;
        stringstream linestream(line);
        long idstatefrom, idstateto, idaction, idoutcome;
        prec_t probability, reward;

        // read idstatefrom
        getline(linestream, cellstring, ',');
        idstatefrom = stoi(cellstring);
        // read idaction
        getline(linestream, cellstring, ',');
        idaction = stoi(cellstring);
        // read idoutcome
        getline(linestream, cellstring, ',');
        idoutcome = stoi(cellstring);
        // read idstateto
        getline(linestream, cellstring, ',');
        idstateto = stoi(cellstring);
        // read probability
        getline(linestream, cellstring, ',');
        probability = stof(cellstring);
        // read reward
        getline(linestream, cellstring, ',');
        reward = stof(cellstring);

        add_transition<Model>(mdp,idstatefrom,idaction,idoutcome,idstateto,probability,reward);

        input >> line;
    }
    return mdp;
}
}

/**
Loads the transition probabilities and rewards from a CSV file.
\param mdp Model output (also returned)
\param filename Name of the file
\param header Whether to create a header of the file too
\returns The input model
 */
template<class Model>
Model& from_csv_file(Model& mdp, const string& filename, bool header = true){
    ifstream ifs(filename);
    from_csv(mdp, ifs, header);
    ifs.close();
    return mdp;
}

/**
Uniformly sets the thresholds to the provided value for all states and actions.
This method should be used only with models that support thresholds.

This function only applies to models that have thresholds, such as ones using
"WeightedOutcomeAction" or its derivatives.

\param model Model to set thresholds for
\param threshold New thresholds value
*/
template<class Model>
void set_thresholds(Model& mdp, prec_t threshold){
    for(auto si : indices(mdp)){
        auto& state = mdp.get_state(si);
        for(auto ai : indices(state)){
            state.get_action(ai).set_threshold(threshold);
        }
    }
}

/**
Checks whether outcome distributions sum to 1 for all states and actions.

This function only applies to models that have thresholds, such as ones using
"WeightedOutcomeAction" or its derivatives.

*/
template<class Model>
bool is_outcomes_normalized(const Model& mdp){
    for(auto si : indices(mdp)){
        auto& state = mdp.get_state(si);
        for(auto ai : indices(state)){
            if(!state.get_action(ai).is_distribution_normalized())
                return false;
        }
    }
    return true;
}

/**
Normalizes outcome distributions for all states and actions.

This function only applies to models that have thresholds, such as ones using
"WeightedOutcomeAction" or its derivatives.
*/
template<class Model>
void normalize_outcomes(Model& mdp){
    for(auto si : indices(mdp)){
        auto& state = mdp.get_state(si);
        for(auto ai : indices(state)){
            state.get_action(ai).normalize_distribution();
        }
    }
}

}

#pragma once

#include "definitions.hpp"
#include "Transition.hpp"
#include "State.hpp"
#include "Action.hpp"
#include "RMDP.hpp"

#include <vector>
#include <istream>
#include <fstream>
#include <memory>
#include <string>
#include <cassert>
#include <sstream>

#include "cpp11-range-master/range.hpp"


// **********************************************************************
// ***********************    HELPER FUNCTIONS    ***********************
// **********************************************************************

namespace craam {

using namespace std;
using namespace util::lang;

/**
Adds a transition probability and reward for a particular outcome.
\param mdp model to add the transition to
\param fromid Starting state ID
\param actionid Action ID
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
*/
template<class Model>
void add_transition(Model& mdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);

/**
Adds a transition probability and reward for a model with no outcomes.
\param mdp model to add the transition to
\param fromid Starting state ID
\param actionid Action ID
\param outcomeid Outcome ID (A single outcome corresponds to a regular MDP)
\param toid Destination ID
\param probability Probability of the transition (must be non-negative)
\param reward The reward associated with the transition.
*/
template<class Model>
void add_transition(Model& mdp, long fromid, long actionid, long toid, prec_t probability, prec_t reward){
    add_transition<Model>(mdp, fromid, actionid, 0l, toid, probability, reward);
}


/**
Loads an GRMDP definition from a simple csv file. States, actions, and
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
Model& from_csv(Model& mdp, istream& input, bool header = true);

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
void set_outcome_thresholds(Model& mdp, prec_t threshold);

/**
Sets the distribution for outcomes for each state and
action to be uniform. 
*/
template<class Model>
void set_uniform_outcome_dst(Model& mdp);

/**
Sets the distribution of outcomes for the given state and action.
*/
template<class Model>
void set_outcome_dst(Model& mdp, size_t stateid, size_t actionid, const numvec& dist);

/**
Checks whether outcome distributions sum to 1 for all states and actions.

This function only applies to models that have thresholds, such as ones using
"WeightedOutcomeAction" or its derivatives.

*/
template<class Model>
bool is_outcome_dst_normalized(const Model& mdp);

/**
Normalizes outcome distributions for all states and actions.

This function only applies to models that have thresholds, such as ones using
"WeightedOutcomeAction" or its derivatives.
*/
template<class Model>
void normalize_outcome_dst(Model& mdp);

/**
Adds uncertainty to a regular MDP. Turns transition probabilities to uncertain
outcomes and uses the transition probabilities as the nominal weights assigned to
the outcomes.

The input is an MDP:
\f$ \mathcal{M} = (\mathcal{S},\mathcal{A},P,r) ,\f$
where the states are \f$ \mathcal{S} = \{ s_1, \ldots, s_n \} \f$
The output RMDP is:
\f$ \bar{\mathcal{M}} = (\mathcal{S},\mathcal{A},\mathcal{B}, \bar{P},\bar{r},d), \f$
where the states and actions are the same as in the original MDP and
\f$ d : \mathcal{S} \times \mathcal{A} \rightarrow \Delta^{\mathcal{B}} \f$ is
the nominal probability of outcomes. Outcomes, transition probabilities, and rewards depend on whether uncertain transitions 
to zero-probability states are allowed:

When allowzeros = true, then \f$ \bar{\mathcal{M}} \f$ will also allow uncertain
transition to states that have zero probabilities in \f$ \mathcal{M} \f$.
- Outcomes are identical for all states and actions:
    \f$ \mathcal{B} = \{ b_1, \ldots, b_n \} \f$
- Transition probabilities are:
    \f$ \bar{P}(s_i,a,b_k,s_l) =  1 \text{ if } k = l, \text{ otherwise } 0  \f$
- Rewards are:
    \f$ \bar{r}(s_i,a,b_k,s_l) = r(s_i,a,s_k) \text{ if } k = l, \text{ otherwise } 0 \f$
- Nominal outcome probabilities are:
    \f$ d(s,a,b_k) = P(s,a,s_k) \f$
    
When allowzeros = false, then \f$ \bar{\mathcal{M}} \f$ will only allow transitions to 
states that have non-zero transition probabilities in \f$ \mathcal{M} \f$. Let \f$ z_k(s,a) \f$ denote 
the \f$ k \f$-th state with a non-zero transition probability from state \f$ s \f$ and action \f$ a \f$.
- Outcomes for \f$ s,a \f$ are:
    \f$ \mathcal{B}(s,a) = \{ b_1, \ldots, b_{|z(s,a)|} \}, \f$
    where \f$ |z(s,a)| \f$ is the number of positive transition probabilities in \f$ P \f$.
- Transition probabilities are:
    \f$ \bar{P}(s_i,a,b_k,s_l) = 1 \text{ if } z_k(s_i,a) = l, \text{ otherwise } 0  \f$
- Rewards are:
    \f$ \bar{r}(s_i,a,b_k,s_k) = r(s_i,a,s_{z_k(s_i,a)}) \f$
- Nominal outcome probabilities are:
    \f$ d(s,a,b_k) = P(s,a,z_k(s,a)) \f$

\tparam SType State type for the RMDP being constructed. The actions must support methods:
    - set_distribution(long outcomeid, prec_t weight)

\param mdp MDP \f$ \mathcal{M} \f$ used as the input
\param allowzeros Whether to allow outcomes to states with zero 
                    transition probability
\returns RMDP with nominal probabilities
*/
template<class SType>
GRMDP<SType> robustify(const MDP& mdp, bool allowzeros);

/**
Instantiated template version of robustify.
*/
RMDP_L1 robustify_l1(const MDP& mdp, bool allowzeros);

}


#include <vector>
#include <istream>
#include <fstream>
#include <memory>
#include <tuple>
#include <cassert>

#include "cpp11-range-master/range.hpp"


/// **********************************************************************
/// ***********************    HELPER FUNCTIONS    ***********************
/// **********************************************************************
namespace craam {

using namespace util::lang;

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


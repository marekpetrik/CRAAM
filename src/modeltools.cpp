#include "modeltools.hpp"

#include "RMDP.hpp"

namespace craam {
    
using namespace util::lang;
   
template<class Model>
void add_transition(Model& mdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){
    // make sure that the destination state exists
    mdp.create_state(toid);
    auto& state_from = mdp.create_state(fromid);
    auto& action = state_from.create_action(actionid);
    Transition& outcome = action.create_outcome(outcomeid);
    outcome.add_sample(toid,probability,reward);
}

template void add_transition<MDP>(MDP& mdp, long fromid, long actionid, 
                        long outcomeid, long toid, prec_t probability, prec_t reward);
template void add_transition<RMDP_D>(RMDP_D& mdp, long fromid, long actionid, 
                        long outcomeid, long toid, prec_t probability, prec_t reward);
template void add_transition<RMDP_L1>(RMDP_L1& mdp, long fromid, long actionid, 
                        long outcomeid, long toid, prec_t probability, prec_t reward);

template<class Model>
Model& from_csv(Model& mdp, istream& input, bool header){
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
        // add transition
        add_transition<Model>(mdp,idstatefrom,idaction,idoutcome,idstateto,probability,reward);
        input >> line;
    }
    return mdp;
}

template MDP& from_csv(MDP& mdp, istream& input, bool header);
template RMDP_D& from_csv(RMDP_D& mdp, istream& input, bool header);
template RMDP_L1& from_csv(RMDP_L1& mdp, istream& input, bool header);


template<class Model>
void set_outcome_thresholds(Model& mdp, prec_t threshold){
    for(const auto si : indices(mdp)){
        auto& state = mdp.get_state(si);
        for(auto ai : indices(state))
            state.get_action(ai).set_threshold(threshold);
    }
}

template void set_outcome_thresholds(RMDP_L1& mdp, prec_t threshold);

template<class Model> void set_uniform_outcome_dst(Model& mdp){

    for(const auto si : indices(mdp)){
        auto& s = mdp[si];
        for(const auto ai : indices(s)){
            auto& a = s[ai];
            numvec distribution(a.size(), 
                    1.0 / static_cast<prec_t>(a.size()));

            a.set_distribution(distribution);
        }
    }
}

template void set_uniform_outcome_dst(RMDP_L1& mdp);

template<class Model> void set_outcome_dst(Model& mdp, size_t stateid, size_t actionid, const numvec& dist){
    assert(stateid >= 0 && stateid < mdp.size());
    assert(actionid >= 0 && actionid < mdp[stateid].size());

    mdp[stateid][actionid].set_distribution(dist);
}

template void set_outcome_dst(RMDP_L1& mdp, size_t stateid, size_t actionid, const numvec& dist);

template<class Model> bool is_outcome_dst_normalized(const Model& mdp){
    for(auto si : indices(mdp)){
        auto& state = mdp.get_state(si);
        for(auto ai : indices(state)){
            if(!state[ai].is_distribution_normalized())
                return false;
        }
    }
    return true;
}

template bool is_outcome_dst_normalized(const RMDP_L1& mdp);

template<class Model> void normalize_outcome_dst(Model& mdp){
    for(auto si : indices(mdp)){
        auto& state = mdp.get_state(si);
        for(auto ai : indices(state))
            state.get_action(ai).normalize_distribution();
    }
}

template void normalize_outcome_dst(RMDP_L1& mdp);

template<class SType>
GRMDP<SType> robustify(const MDP& mdp, bool allowzeros){
    // construct the result first
    GRMDP<SType> rmdp;
    // iterate over all starting states (at t)
    for(size_t si : indices(mdp)){
        const auto& s = mdp[si];
        auto& newstate = rmdp.create_state(si);
        for(size_t ai : indices(s)){
            auto& newaction = newstate.create_action(ai);
            const Transition& t = s[ai].get_outcome();
            // iterate over transitions next states (at t+1) and add samples
            if(allowzeros){
                numvec probabilities = t.probabilities_vector(mdp.state_count());
                numvec rewards = t.rewards_vector(mdp.state_count());
                for(size_t nsi : indices(probabilities)){
                    // create the outcome with the appropriate weight
                    Transition& newoutcome = 
                        newaction.create_outcome(newaction.size(), 
                                                probabilities[nsi]);
                    // adds the single sample for each outcome
                    newoutcome.add_sample(nsi, 1.0, rewards[nsi]);
                }    
            }
            else{
                // only consider non-zero probabilities unless allowzeros is used
                for(size_t nsi : indices(t)){
                    // create the outcome with the appropriate weight
                    Transition& newoutcome = 
                        newaction.create_outcome(newaction.size(), 
                                                t.get_probabilities()[nsi]);
                    // adds the single sample for each outcome
                    newoutcome.add_sample(t.get_indices()[nsi], 1.0, t.get_rewards()[nsi]);
                }    
            }
        }
    }    
    return rmdp;
}

RMDP_L1 robustify_l1(const MDP& mdp, bool allowzeros){
    return robustify<L1RobustState>(mdp, allowzeros);
}
// -----------------------------------
// Specific template instantiations
// -----------------------------------

template RMDP_L1 robustify<L1RobustState>(const MDP&, bool);

}

#include "modeltools.hpp"

#include "RMDP.hpp"

namespace craam {
    
using namespace util::lang;
    
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

// -----------------------------------
// Specific template instantiations
// -----------------------------------

template RMDP_L1 robustify<L1RobustState>(const MDP&, bool);

}

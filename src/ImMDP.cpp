#include "ImMDP.hpp"

namespace craam{
namespace impl{


ImMDP::ImMDP() {
}


unique_ptr<RMDP> to_robust_mdp(){
    /**
        Constructs a robust version of the MDP.

        The new MDP has a single state for each observation, and the
        original states are treated as outcomes. The actions are preserved.
    */

}

}}


#pragma once

#include "RMDP.hpp"
#include <vector>
#include <memory>

using namespace std;

namespace craam{
namespace impl{

class ImMDP{
/**
  Markov Decision Process with implementability constraints on
  taking the same actions in multiple states.

  The states in the same "observation" must have the same action.
*/
public:

    /** The internal representation of the MDP */
    RMDP mdp;
    /** Represents the index of the observations */
    vector<int> observations;

    ImMDP();

    unique_ptr<RMDP> to_robust_mdp();

    bool check_correctness();

protected:
    /** Robust MDP version of the implementable MDP */
    unique_ptr<RMDP> robust_mdp;

};

}}

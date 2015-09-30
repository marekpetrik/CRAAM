#pragma once

#include "RMDP.hpp"
#include "Transition.hpp"

#include <vector>
#include <memory>

using namespace std;

namespace craam{
namespace impl{


class IMDP{
/**
    Represents an MDP with implementability constraints

    Consists of an MDP and a set of observations.
*/

public:
    const shared_ptr<const RMDP> mdp;
    const vector<long> observations;
    const Transition initial;


    IMDP(const shared_ptr<const RMDP>& mdp, const vector<long>& observations,
            const Transition& initial) :
        mdp(mdp), observations(observations), initial(initial)
    {
        /**
            \param mdp A non-robust base MDP model
            \param observations Maps each state to the index of the corresponding observation.
                            A valid policy will take the same action in all states
                            with a single observation. The index is 0-based.
            \param initial A representation of the initial distribution. The rewards
                            in this transition are ignored (and should be 0).
        */

    };

    unique_ptr<RMDP> to_robust();

};


}}

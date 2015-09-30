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
        mdp(mdp), observations(observations), initial(initial);

    unique_ptr<RMDP> to_robust();

};


}}

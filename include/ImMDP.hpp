#pragma once

#include "RMDP.hpp"
#include "Transition.hpp"

#include <vector>
#include <memory>

using namespace std;

namespace craam{
namespace impl{

/**
    Represents an MDP with implementability constraints

    Consists of an MDP and a set of observations.
*/
class MDPI{

public:
    const shared_ptr<const RMDP> mdp;
    const vector<long> observ2state;
    const Transition initial;


    MDPI(const shared_ptr<const RMDP>& mdp, const vector<long>& observ2state, const Transition& initial);

    MDPI(const RMDP& mdp, const vector<long>& observ2state, const Transition& initial);
    
protected:
    static void check_parameters(const RMDP& mdp, const vector<long>& observ2state, const Transition& initial);
};


/**
    An MDP with implementability constraints. The class contains solution
    methods that rely on robust MDP reformulation of the problem.
 */
class MDPI_R : public MDPI{

public:


    MDPI_R(const shared_ptr<const RMDP>& mdp, const vector<long>& observ2state,
            const Transition& initial);

    MDPI_R(const RMDP& mdp, const vector<long>& observ2state, const Transition& initial);

    const RMDP& get_robust_mdp() const {
        /** Returns the internal robust MDP representation  */
        return robust_mdp;
    };

protected:


    RMDP robust_mdp;    // TODO: this would ideally be a constant
    vector<long> state2outcome;

    void initialize_robustmdp();
};

}}

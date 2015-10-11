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
    MDPI(const shared_ptr<const RMDP>& mdp, const indvec& state2observ, const Transition& initial);

    MDPI(const RMDP& mdp, const indvec& state2observ, const Transition& initial);

    size_t obs_count() const { return maxobs };
    size_t state_count() const {return mdp->state_count()};

    indvec obspol_to_statepol(indvec obspol);
    
protected:
    shared_ptr<const RMDP> mdp;
    indvec state2observ;
    long maxobs;
    Transition initial;

    static void check_parameters(const RMDP& mdp, const indvec& state2observ, const Transition& initial);
};


/**
    An MDP with implementability constraints. The class contains solution
    methods that rely on robust MDP reformulation of the problem.
 */
class MDPI_R : public MDPI{

public:

    MDPI_R(const shared_ptr<const RMDP>& mdp, const indvec& state2observ,
            const Transition& initial);

    MDPI_R(const RMDP& mdp, const indvec& state2observ, const Transition& initial);

    const RMDP& get_robust_mdp() const {
        /** Returns the internal robust MDP representation  */
        return robust_mdp;
    };

    Solution solve_reweighted(long iterations) const;

protected:

    RMDP robust_mdp;    // TODO: this would ideally be a constant
    indvec state2outcome;

    void initialize_robustmdp();
};

}}

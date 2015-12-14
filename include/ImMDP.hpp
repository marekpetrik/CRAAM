#pragma once

#include "RMDP.hpp"
#include "Transition.hpp"

#include <vector>
#include <memory>

using namespace std;

namespace craam{
namespace impl{

/**
    Represents an MDP with implementability constraints.

    Consists of an MDP and a set of observations.
*/
class MDPI{

public:
    MDPI(const shared_ptr<const RMDP>& mdp, const indvec& state2observ,
         const Transition& initial);

    MDPI(const RMDP& mdp, const indvec& state2observ, const Transition& initial);

    size_t obs_count() const { return obscount; };
    size_t state_count() const {return mdp->state_count(); };
    long state2obs(long state){return state2observ[state];};
    size_t action_count(long obsid) {return action_counts[obsid];};

    indvec obspol2statepol(indvec obspol) const;

    shared_ptr<const RMDP> get_mdp() {return mdp;};
    Transition get_initial() const {return initial;};

    // save and load description.
    void to_csv(ostream& output_mdp, ostream& output_state2obs, ostream& output_initial,
                    bool headers = true) const;
    void to_csv_file(const string& output_mdp, const string& output_state2obs,
                     const string& output_initial, bool headers = true) const;

    template<typename T = MDPI>
    static unique_ptr<T> from_csv(istream& input_mdp, istream& input_state2obs,
                                     istream& input_initial, bool headers = true);
    template<typename T = MDPI>
    static unique_ptr<T> from_csv_file(const string& input_mdp,
                                          const string& input_state2obs,
                                          const string& input_initial,
                                          bool headers = true);

    indvec random_policy(random_device::result_type seed = random_device{}());

protected:

    /** the underlying MDP */
    shared_ptr<const RMDP> mdp;
    /** maps index of a state to the index of the observation */
    indvec state2observ;
    /** initial distribution */
    Transition initial;
    /** number of observations */
    long obscount;
    /** number of actions for each observation */
    indvec action_counts;

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

    void update_importance_weights(const numvec& weights);
    indvec solve_reweighted(long iterations, prec_t discount);

    static unique_ptr<MDPI_R> from_csv(istream& input_mdp, istream& input_state2obs,
                                     istream& input_initial, bool headers = true){

        return MDPI::from_csv<MDPI_R>(input_mdp,input_state2obs,input_initial, headers);
    };

    static unique_ptr<MDPI_R> from_csv_file(const string& input_mdp,
                                          const string& input_state2obs,
                                          const string& input_initial,
                                          bool headers = true){
        return MDPI::from_csv_file<MDPI_R>(input_mdp,input_state2obs,input_initial, headers);
    };

protected:
    /** the robust representation of the MDPI */
    RMDP robust_mdp;
    /** maps the index of the mdp state to the index of the observation
        withing the state corresponding to the observation */
    indvec state2outcome;

    void initialize_robustmdp();
};

}}

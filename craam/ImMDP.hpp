// This file is part of CRAAM, a C++ library for solving plain
// and robust Markov decision processes.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of
// this software and associated documentation files (the "Software"), to deal in
// the Software without restriction, including without limitation the rights to
// use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
// the Software, and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#pragma once

#include "craam/RMDP.hpp"
#include "craam/Transition.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"
#include "craam/algorithms/robust_values.hpp"
#include "craam/algorithms/nature_response.hpp"
#include "craam/algorithms/occupancies.hpp"

#include <rm/range.hpp>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>
#include <iostream>
#include <iterator>


namespace craam{
/// A namespace with tools for implementable, interpretable, and aggregated MDPs
namespace impl{

using namespace std;
using namespace util::lang;
using namespace craam::algorithms;

template<typename T>
T max_value(vector<T> x){
    return (x.size() > 0) ? *max_element(x.begin(), x.end()) : -1;
}

/**
Represents an MDP with implementability constraints.

Consists of an MDP and a set of observations.
*/
class MDPI{

public:
    /**
    Constructs the MDP with implementability constraints. This constructor makes it
    possible to share the MDP with other data structures.

    Important: when the underlying MDP changes externally, the object becomes invalid
    and may result in unpredictable behavior.

    \param mdp A non-robust base MDP model.
    \param state2observ Maps each state to the index of the corresponding observation.
                    A valid policy will take the same action in all states
                    with a single observation. The index is 0-based.
    \param initial A representation of the initial distribution. The rewards
                    in this transition are ignored (and should be 0).
    */
    MDPI(const shared_ptr<const MDP>& mdp, const indvec& state2observ, const Transition& initial):
             mdp(mdp), state2observ(state2observ), initial(initial),
              obscount(1+max_value(state2observ)),
              action_counts(obscount, -1){
        check_parameters(*mdp, state2observ, initial);

        for(auto state : range((size_t) 0, mdp->state_count())){
            auto obs = state2observ[state];

            // check the number of actions
            auto ac = mdp->get_state(state).action_count();
            if(action_counts[obs] >= 0){
                if(action_counts[obs] != (long) ac){
                    throw invalid_argument("Inconsistent number of actions: " + to_string(ac) +
                                           " instead of " + to_string(action_counts[obs]) +
                                           " in state " + to_string(state));
                }
            }else{
                action_counts[obs] = ac;
            }
        }
    }

    /**
    Constructs the MDP with implementability constraints. The MDP model is
    copied (using the copy constructor) and stored internally.
    \param mdp A non-robust base MDP model. It cannot be shared to prevent
                direct modification.
    \param state2observ Maps each state to the index of the corresponding observation.
                    A valid policy will take the same action in all states
                    with a single observation. The index is 0-based.
    \param initial A representation of the initial distribution. The rewards
                    in this transition are ignored (and should be 0).
    */
    MDPI(const MDP& mdp, const indvec& state2observ, const Transition& initial)
            : MDPI(make_shared<const MDP>(mdp),state2observ, initial){}

    size_t obs_count() const { return obscount; };
    size_t state_count() const {return mdp->state_count(); };
    long state2obs(long state){return state2observ[state];};
    size_t action_count(long obsid) {return action_counts[obsid];};

    /**
    Converts a policy defined in terms of observations to a policy defined in
    terms of states.
    \param obspol Policy that maps observations to actions to take
    \return Observation policy
    */
    indvec obspol2statepol(const indvec& obspol) const{
        indvec statepol(state_count());
        obspol2statepol(obspol, statepol);
        return statepol;
    }

    /**
    Converts a policy defined in terms of observations to a policy defined in
    terms of states.
    \param obspol Policy that maps observations to actions to take
    \param statepol State policy target
    */
    void obspol2statepol(const indvec& obspol, indvec& statepol) const{
        assert(obspol.size() == (size_t) obscount);
        assert(mdp->state_count() == statepol.size());

        for(auto s : range((size_t)0, state_count())){
            statepol[s] = obspol[state2observ[s]];
        }
    }

    /**
    Converts a transition from states to observations, adding probabilities
    of individual states. Rewards are a convex combination of the original
    values.
    */
    Transition transition2obs(const Transition& tran){
        if((size_t) tran.max_index() >= state_count())
        throw invalid_argument("Transition to a non-existing state.");
        Transition result;
        for(auto i : range((size_t)0, tran.size())){
            const long state = tran.get_indices()[i];
            const prec_t prob = tran.get_probabilities()[i];
            const prec_t reward = tran.get_rewards()[i];

            result.add_sample(state2obs(state), prob, reward);
        }
        return result;
    }

    /** Internal MDP representation */
    shared_ptr<const MDP> get_mdp() {return mdp;};

    /** Initial distribution of MDP */
    Transition get_initial() const {return initial;};

    /** Constructs a random observation policy */
    indvec random_policy(random_device::result_type seed = random_device{}()){
        indvec policy(obscount, -1);

        default_random_engine gen(seed);

        for(auto obs : range(0l, obscount)){
            auto ac = action_counts[obs];
            if(ac == 0)
                continue;

            uniform_int_distribution<int> dist(0,ac-1);
            policy[obs] = dist(gen);
        }

        return policy;
    }

    /**
    Computes a return of an observation policy.

    \param discount Discount factor
    \return Discounted return of the policy
    */
    prec_t total_return(prec_t discount, prec_t precision=SOLPREC) const{
        auto&& sol = mpi_jac(*mdp, discount, numvec(0), PlainBellman<RegularState>(), MAXITER, precision);
        return sol.total_return(initial);
    }

    // save and load description.
    /**
    Saves the MDPI to a set of 3 csv files, for transitions,
    observations, and the initial distribution

    \param output_mdp Transition probabilities and rewards
    \param output_state2obs Mapping states to observations
    \param output_initial Initial distribution
    */
    void to_csv(ostream& output_mdp, ostream& output_state2obs, ostream& output_initial,
                    bool headers = true) const{
        // save the MDP
        craam::to_csv(*mdp, output_mdp, headers);
        // save state maps
        if(headers){
            output_state2obs << "idstate,idobs" << endl;
        }
        for(auto i : indices(state2observ)){
            output_state2obs << i << "," << state2observ[i] << endl;
        }

        // save the initial distribution
        if(headers){
            output_initial << "idstate,probability" << endl;
        }
        const indvec& inindices = initial.get_indices();
        const numvec& probabilities = initial.get_probabilities();

        for(auto i : indices(inindices)){
            output_initial << inindices[i] << "," << probabilities[i] << endl;
        }
    
    }

    /**
    Saves the MDPI to a set of 3 csv files, for transitions, observations,
    and the initial distribution

    \param output_mdp File name for transition probabilities and rewards
    \param output_state2obs File name for mapping states to observations
    \param output_initial File name for initial distribution
    */
    void to_csv_file(const string& output_mdp, const string& output_state2obs,
                     const string& output_initial, bool headers = true) const{
    
        // open file streams
        ofstream ofs_mdp(output_mdp),
                    ofs_state2obs(output_state2obs),
                    ofs_initial(output_initial);

        // save the data
        to_csv(ofs_mdp, ofs_state2obs, ofs_initial, headers);

        // close streams
        ofs_mdp.close(); ofs_state2obs.close(); ofs_initial.close();
    }

    /**
    Loads an MDPI from a set of 3 csv files, for transitions, observations,
    and the initial distribution

    The MDP size is defined by the transitions file.

    \param input_mdp File name for transition probabilities and rewards
    \param input_state2obs File name for mapping states to observations
    \param input_initial File name for initial distribution
     */
    template<typename T = MDPI>
    static unique_ptr<T> from_csv(istream& input_mdp, istream& input_state2obs,
                                     istream& input_initial, bool headers = true){
        // read mdp
        MDP mdp;
        craam::from_csv(mdp,input_mdp);

        // read state2obs
        string line;
        if(headers) input_state2obs >> line; // skip the header

        indvec state2obs(mdp.state_count());
        input_state2obs >> line;
        while(input_state2obs.good()){
            string cellstring;
            stringstream linestream(line);

            getline(linestream, cellstring, ',');
            auto idstate = stoi(cellstring);
            getline(linestream, cellstring, ',');
            auto idobs = stoi(cellstring);
            state2obs[idstate] = idobs;

            input_state2obs >> line;
        }

        // read initial distribution
        if(headers) input_initial >> line; // skip the header

        Transition initial;
        input_initial >> line;
        while(input_initial.good()){
            string cellstring;
            stringstream linestream(line);

            getline(linestream, cellstring, ',');
            auto idstate = stoi(cellstring);
            getline(linestream, cellstring, ',');
            auto prob = stof(cellstring);
            initial.add_sample(idstate, prob, 0.0);

            input_initial >> line;
        }

        shared_ptr<const MDP> csmdp = make_shared<const MDP>(std::move(mdp));
        return make_unique<T>(csmdp, state2obs, initial);
        
    }
    template<typename T = MDPI>
    static unique_ptr<T> from_csv_file(const string& input_mdp,
                                          const string& input_state2obs,
                                          const string& input_initial,
                                          bool headers = true){
        // open files
        ifstream ifs_mdp(input_mdp),
                    ifs_state2obs(input_state2obs),
                    ifs_initial(input_initial);

        // transfer method call
        return from_csv<T>(ifs_mdp, ifs_state2obs, ifs_initial, headers);
    }
protected:

    /** the underlying MDP */
    shared_ptr<const MDP> mdp;
    /** maps index of a state to the index of the observation */
    indvec state2observ;
    /** initial distribution */
    Transition initial;
    /** number of observations */
    long obscount;
    /** number of actions for each observation */
    indvec action_counts;

    /**
     Checks whether the parameters are correct. Throws an exception if the parameters
     are wrong.
     */
    static void check_parameters(const MDP& mdp, const indvec& state2observ, const Transition& initial){
        // *** check consistency of provided parameters ***
        // check that the number of state2observ coefficients it correct
        if(mdp.state_count() !=  state2observ.size())
            throw invalid_argument("Number of observation indexes must match the number of states.");
        // check that the observation indexes are not negative
        if(state2observ.size() == 0)
            throw invalid_argument("Cannot have empty observations.");
        if(*min_element(state2observ.begin(), state2observ.end()) < 0)
            throw invalid_argument("Observation indexes must be non-negative");
        // check then initial transition
        if(initial.max_index() >= (long) mdp.state_count())
            throw invalid_argument("An initial transition to a non-existent state.");
        if(!initial.is_normalized())
            throw invalid_argument("The initial transition must be normalized.");
    }
};


/**
An MDP with implementability constraints. The class contains solution
methods that rely on robust MDP reformulation of the problem.
 */
class MDPI_R : public MDPI{

public:

    /**
    Calls the base constructor and also constructs the corresponding
    robust MDP
     */
    MDPI_R(const shared_ptr<const MDP>& mdp, const indvec& state2observ, const Transition& initial) 
        : MDPI(mdp, state2observ, initial), robust_mdp(), state2outcome(mdp->state_count(),-1){
        initialize_robustmdp();
    }

    /**
    Calls the base constructor and also constructs the corresponding
    robust MDP.
    */
    MDPI_R(const MDP& mdp, const indvec& state2observ, const Transition& initial)
        : MDPI(mdp, state2observ, initial), robust_mdp(), state2outcome(mdp.state_count(),-1){
        initialize_robustmdp();
    }

    const RMDP& get_robust_mdp() const {
        /** Returns the internal robust MDP representation  */
        return robust_mdp;
    };

    /**
    Updates the weights on outcomes in the robust MDP based on the state
    weights provided.

    This method modifies the stored robust MDP.
     */
    void update_importance_weights(const numvec& weights){
        if(weights.size() != state_count()){
            throw invalid_argument("Size of distribution must match the number of states.");
        }

        // loop over all mdp states and set weights
        for(size_t i : indices(weights)){
            const auto rmdp_stateid = state2observ[i];
            const auto rmdp_outcomeid = state2outcome[i];

            // loop over all actions
            auto& rstate = robust_mdp.get_state(rmdp_stateid);
            for(size_t ai : indices(rstate)){
                rstate.get_action(ai).set_distribution(rmdp_outcomeid, weights[i]);
            }
        }

        // now normalize the weights to they sum to one
        for(size_t si : indices(robust_mdp)){
            auto& s = robust_mdp.get_state(si);
            for(size_t ai : indices(s)){
                auto& a = s.get_action(ai);
                // check if the distribution sums to 0 (not visited)
                const numvec& dist = a.get_distribution();
                if(accumulate(dist.begin(), dist.end(), 0.0) > 0.0){
                    a.normalize_distribution();
                }
                else{
                    // just set it to be uniform
                    a.uniform_distribution();
                }
            }
        }
    }

    /**
    Uses a simple iterative algorithm to solve the MDPI.

    The algorithm starts with a policy composed of actions all 0, and
    then updates the distribution of robust outcomes (corresponding to MDP states),
    and computes the optimal solution for thus weighted RMDP.

    This method modifies the stored robust MDP.

    \param iterations Maximal number of iterations; terminates when the policy no longer changes
    \param discount Discount factor
    \param initobspol Initial observation policy (optional). When omitted or has length 0
        a policy that takes the first action (action 0) is used.
    \returns Policy for observations (an index of each action for each observation)
    */
    indvec solve_reweighted(long iterations, prec_t discount, const indvec& initobspol = indvec(0)){
        if(initobspol.size() > 0 && initobspol.size() != obs_count()){
            throw invalid_argument("Initial policy must be defined for all observations.");
        }

        indvec obspol(initobspol);                   // return observation policy
        if(obspol.size() == 0){
            obspol.resize(obs_count(),0);
        }
        indvec statepol(state_count(),0);         // state policy that corresponds to the observation policy
        obspol2statepol(obspol,statepol);

        // map the initial distribution to observations in order to evaluate the return
        const Transition oinitial = transition2obs(initial);

        for(auto iter : range(0l, iterations)){
            (void) iter; // to remove the warning

            // compute state distribution
            numvec importanceweights = occfreq_mat(*mdp, initial, discount, statepol);
            // update importance weights
            update_importance_weights(importanceweights);
            // compute solution of the robust MDP with the new weights
            auto&& s = mpi_jac(robust_mdp, discount);

            // update the policy for the underlying states
            obspol = s.policy;
            // map the observation policy to the individual states
            obspol2statepol(obspol, statepol);
        }
        return obspol;
    }

    /**
    Uses a robust MDP formulation to solve the MDPI. States in the observation are treated
    as outcomes. The baseline distribution is inferred from the provided policy.

    The uncertainty is bounded by using an L1 norm deviation and the provided
    threshold.

    The method can run for several iterations, like solve_reweighted.

    \param iterations Maximal number of iterations; terminates when the policy no longer changes
    \param threshold Upper bound on the L1 deviation from the baseline distribution.
    \param discount Discount factor
    \param initobspol Initial observation policy (optional). When omitted or has length 0
        a policy that takes the first action (action 0) is used.
    \returns Policy for observations (an index of each action for each observation)
    */
    indvec solve_robust(long iterations, prec_t threshold, prec_t discount, const indvec& initobspol = indvec(0)){
    
        if(initobspol.size() > 0 && initobspol.size() != obs_count()){
            throw invalid_argument("Initial policy must be defined for all observations.");
        }

        indvec obspol(initobspol);                   // return observation policy
        if(obspol.size() == 0){
            obspol.resize(obs_count(),0);
        }
        indvec statepol(state_count(),0);         // state policy that corresponds to the observation policy
        obspol2statepol(obspol,statepol);

        for(auto iter : range(0l, iterations)){
            (void) iter; // to remove the warning

            // compute state distribution
            numvec&& importanceweights = occfreq_mat(*mdp, initial, discount, statepol);

            // update importance weights
            update_importance_weights(importanceweights);

            // compute solution of the robust MDP with the new weights
            auto bu = SARobustBellman<WeightedRobustState>(nats::robust_l1u(threshold));
            auto&& s = mpi_jac(robust_mdp, discount, numvec(0), bu);

            // update the policy for the underlying states
            obspol = unzip(s.policy).first;

            // map the observation policy to the individual states
            obspol2statepol(obspol, statepol);

        }
        return obspol;
    
    }

    static unique_ptr<MDPI_R> from_csv(istream& input_mdp, istream& input_state2obs,
                                     istream& input_initial, bool headers = true){

        return MDPI::from_csv<MDPI_R>(input_mdp,input_state2obs,input_initial, headers);
    };

    /** Loads the class from an set of CSV files. See also from_csv. */
    static unique_ptr<MDPI_R> from_csv_file(const string& input_mdp,
                                          const string& input_state2obs,
                                          const string& input_initial,
                                          bool headers = true){
        return MDPI::from_csv_file<MDPI_R>(input_mdp,input_state2obs,input_initial, headers);
    };

protected:
    /** Robust representation of the MDPI */
    RMDP robust_mdp;
    /** Maps the index of the mdp state to the index of the observation
    within the state corresponding to the observation (multiple states per observation) */
    indvec state2outcome;
    /** Constructs a robust version of the implementable MDP.*/
    void initialize_robustmdp(){
        // Determine the number of state2observ
        auto obs_count = *max_element(state2observ.begin(), state2observ.end()) + 1;

        // keep track of the number of outcomes for each
        indvec outcome_count(obs_count, 0);

        for(size_t state_index : indices(*mdp)){
            auto obs = state2observ[state_index];

            // make sure to at least create a terminal state when there are no actions for it
            robust_mdp.create_state(obs);

            // maps the transitions
            for(auto action_index : range(0l, action_counts[obs])){
                // get original MDP transition
                const Transition& old_tran = mdp->get_state(state_index).get_action(action_index).get_outcome();
                // create a new transition
                Transition& new_tran = robust_mdp.create_state(obs).create_action(action_index).create_outcome(outcome_count[obs]);

                // copy the original transitions (they are automatically consolidated while being added)
                for(auto k : range((size_t) 0, old_tran.size())){
                    new_tran.add_sample(state2observ[old_tran.get_indices()[k]],
                                        old_tran.get_probabilities()[k],
                                        old_tran.get_rewards()[k]);
                }

            }
            state2outcome[state_index] = outcome_count[obs]++;
        }
    }
};

}}

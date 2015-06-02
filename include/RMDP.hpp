#ifndef RMDP_H
#define RMDP_H

#include <vector>
#include <istream>
#include <ostream>
#include <fstream>
#include <memory>

#include "definitions.hpp"
#include "State.hpp"

using namespace std;

enum Uncertainty {
    simplex = 0
};

enum SolutionType {
    Robust = 0,
    Optimistic = 1,
    Average = 2
};

class Solution {
public:
    vector<prec_t> valuefunction;
    vector<long> policy;                        // index of the actions for each states
    vector<long> outcomes;                      // index of the outcome for each state
    vector<vector<prec_t>> outcome_dists;       // distribution of outcomes for each state
    prec_t residual;
    long iterations;

    Solution(vector<prec_t> const& valuefunction, vector<long> const& policy, vector<long> const& outcomes){
        this->valuefunction = valuefunction;
        this->policy = policy;
        this->outcomes = outcomes;
    };

    Solution(vector<prec_t> const& valuefunction, vector<long> const& policy, vector<vector<prec_t>> const& outcome_dists){
        this->valuefunction = valuefunction;
        this->policy = policy;
        this->outcome_dists = outcome_dists;
    };

    Solution(vector<prec_t> const& valuefunction, vector<long> const& policy, vector<long> const& outcomes, prec_t residual, long iterations){
        this->valuefunction = valuefunction;
        this->policy = policy;
        this->outcomes = outcomes;
        this->residual = residual;
        this->iterations = iterations;
    };

    Solution(vector<prec_t> const& valuefunction, vector<long> const& policy, vector<vector<prec_t>> const& outcome_dists, prec_t residual, long iterations){
        this->valuefunction = valuefunction;
        this->policy = policy;
        this->outcome_dists = outcome_dists;
        this->residual = residual;
        this->iterations = iterations;
    };

    Solution(){
    };
};

class RMDP{
/**
 * Some general assumptions:
 *
 *    * 0-probability transitions may be omitted
 *    * state with no actions: a terminal state with value 0
 *    * action with no outcomes: terminates with an error
 *    * outcome with no target states: terminates with an error
 */


public:
    vector<State> states;


    RMDP(long state_count){
        /** \brief Constructs the RMDP with a pre-allocated number of states.
         *
         * The state ids must be sequential and are constructed as needed.
         *
         * \param state_count The number of states.
         */

        states = vector<State>(state_count);
    };

    RMDP() : RMDP(0) {};

    // adding transitions
    void add_transition(long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward);
    void add_transition_d(long fromid, long actionid, long toid, prec_t probability, prec_t reward);
    void add_transitions(vector<long> const& fromids, vector<long> const& actionids, vector<long> const& outcomeids, vector<long> const& toids, vector<prec_t> const& probs, vector<prec_t> const& rews);

    // manipulate MDP attributes
    void set_distribution(long fromid, long actionid, vector<prec_t> const& distribution, prec_t threshold);
    void set_threshold(long stateid, long actionid, prec_t threshold);
    void set_uniform_distribution(prec_t threshold);
    void set_uniform_thresholds(prec_t threshold);
    void set_reward(long stateid, long actionid, long outcomeid, long sampleid, prec_t reward);

    // querying parameters
    prec_t get_reward(long stateid, long actionid, long outcomeid, long sampleid) const;
    Transition& get_transition(long stateid, long actionid, long outcomeid);
    const Transition& get_transition(long stateid, long actionid, long outcomeid) const;
    prec_t get_threshold(long stateid, long actionid) const;

    // object counts
    long state_count() const;
    long action_count(long stateid) const;
    long outcome_count(long stateid, long actionid) const;
    long transition_count(long stateid, long actionid, long outcomeid) const;
    long sample_count(long stateid, long actionid, long outcomeid) const;

    // normalization
    bool is_normalized() const;
    void normalize();

    // value iteration
    Solution vi_gs(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual, SolutionType type) const;
    Solution vi_gs_l1(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual, SolutionType type) const;

    Solution vi_jac(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual, SolutionType type) const;
    Solution vi_jac_l1(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual, SolutionType type) const;

    // modified policy iteration
    Solution mpi_jac(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi, SolutionType type) const;
    Solution mpi_jac_l1(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi, SolutionType type) const;


    // writing a reading files
    static unique_ptr<RMDP> transitions_from_csv(istream& input, bool header = true);
    void transitions_to_csv(ostream& output, bool header = true) const;
    void transitions_to_csv_file(const string& filename, bool header = true) const;

    // string representation
    string to_string() const;
};

#endif // RMDP_H

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

namespace craam {

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
  Some general assumptions:

     * 0-probability transitions may be omitted
     * state with no actions: a terminal state with value 0
     * action with no outcomes: terminates with an error
     * outcome with no target states: terminates with an error
 */


public:
    vector<State> states;


    RMDP(long state_count){
        /** Constructs the RMDP with a pre-allocated number of states.

          The state ids must be sequential and are constructed as needed.

          \param state_count The number of states.
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
    prec_t get_toid(long stateid, long actionid, long outcomeid, long sampleid) const;
    prec_t get_probability(long stateid, long actionid, long outcomeid, long sampleid) const;
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
    template<SolutionType type>
    Solution vi_gs(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallellized). The outcomes are
           selected using worst-case optimization.

           Because this function updates the array value furing the iteration, it may be
           difficult to prallelize easily.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
           \param type Whether the solution is maximized or minimized or computed average
         */

        if(valuefunction.size() != states.size()){
            throw invalid_argument("incorrect size of value function");
        }

        vector<long> policy(states.size());
        vector<long> outcomes(states.size());

        prec_t residual = numeric_limits<prec_t>::infinity();
        size_t i;

        for(i = 0; i < iterations && residual > maxresidual; i++){
            residual = 0;

            for(size_t s = 0l; s < states.size(); s++){
                const auto& state = states[s];

                pair<long,prec_t> avgvalue;
                tuple<long,long,prec_t> newvalue;

                switch(type){
                case SolutionType::Robust:
                    newvalue = state.max_min(valuefunction,discount);
                    break;
                case SolutionType::Optimistic:
                    newvalue = state.max_max(valuefunction,discount);
                    break;
                case SolutionType::Average:
                    avgvalue = state.max_average(valuefunction,discount);
                    // TODO replace by make_tuple
                    newvalue = tuple<long,long,prec_t>(avgvalue.first,-1,avgvalue.second);
                    break;
                }

                residual = max(residual, abs(valuefunction[s] - get<2>(newvalue)));
                valuefunction[s] = get<2>(newvalue);

                policy[s] = get<0>(newvalue);
                outcomes[s] = get<1>(newvalue);
            }
        }
        return Solution(valuefunction,policy,outcomes,residual,i);
    };

    template<SolutionType type, pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    Solution vi_gs_cst(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallellized). The outcomes are
           selected using l1 bounds, as specified by set_distribution.

           Because this function updates the array value furing the iteration, it may be
           difficult to prallelize easily.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
           \param type Whether the solution is maximized or minimized or computed average
         */


        if(valuefunction.size() != this->states.size()){
            throw invalid_argument("incorrect size of value function");
        }

        vector<long> policy(this->states.size());
        vector<vector<prec_t>> outcome_dists(this->states.size());

        prec_t residual = numeric_limits<prec_t>::infinity();
        size_t i;

        for(i = 0; i < iterations && residual > maxresidual; i++){

            residual = 0;
            for(auto s=0l; s < (long) this->states.size(); s++){
                const auto& state = this->states[s];

                tuple<long,vector<prec_t>,prec_t> newvalue;
                switch(type){
                case SolutionType::Robust:
                    newvalue = state.max_min_l1(valuefunction, discount);
                    break;
                case SolutionType::Optimistic:
                    newvalue = state.max_max_l1(valuefunction, discount);
                    break;
                default:
                    throw invalid_argument("unknown/invalid (average not supported) optimization type.");
                }
                residual = max(residual, abs(valuefunction[s] - get<2>(newvalue) ));
                valuefunction[s] = get<2>(newvalue);
                outcome_dists[s] = get<1>(newvalue);
                policy[s] = get<0>(newvalue);
            }
        }
        return Solution(valuefunction,policy,outcome_dists,residual,i);
    };

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

    // copying
    unique_ptr<RMDP> copy() const;
    void copy_into(RMDP& result) const;

    // string representation
    string to_string() const;
};
};

#endif // RMDP_H

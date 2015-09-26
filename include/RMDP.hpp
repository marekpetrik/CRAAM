#pragma once
#include <vector>
#include <istream>
#include <ostream>
#include <fstream>
#include <memory>
#include <cmath>
#include <tuple>
#include <algorithm>

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
        /**
          Constructs the RMDP with a pre-allocated number of states. The state ids
          must be sequential and are constructed as needed.

          \param state_count The number of states.
         */

        states = vector<State>(state_count);
    };

    RMDP() : RMDP(0) {
        /**
          Constructs an empty RMDP.
         */

        states = vector<State>(0);
    
        };

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
    Solution vi_gs_gen(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    Solution vi_gs_rob(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallelized). The outcomes are
           selected using worst-case nature.

           This function is suitable for computing the value function of a finite state MDP. If
           the states are ordered correctly, one iteration is enough to compute the optimal value function.
           Since the value function is updated from the first state to the last, the states should be ordered
           in reverse temporal order.

           Because this function updates the array value during the iteration, it may be
           difficult to parallelize easily.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_gen<SolutionType::Robust>(valuefunction, discount, iterations, maxresidual);
    };

    Solution vi_gs_opt(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallelized). The outcomes are
           selected using best-case nature.

           This function is suitable for computing the value function of a finite state MDP. If
           the states are ordered correctly, one iteration is enough to compute the optimal value function.
           Since the value function is updated from the first state to the last, the states should be ordered
           in reverse temporal order.

           Because this function updates the array value during the iteration, it may be
           difficult to parallelize easily.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_gen<SolutionType::Optimistic>(valuefunction, discount, iterations, maxresidual);
    };

    Solution vi_gs_ave(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallelized). The outcomes are
           selected using average-case nature.

           This function is suitable for computing the value function of a finite state MDP. If
           the states are ordered correctly, one iteration is enough to compute the optimal value function.
           Since the value function is updated from the first state to the last, the states should be ordered
           in reverse temporal order.

           Because this function updates the array value during the iteration, it may be
           difficult to paralelize easily.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
    };

    template<SolutionType type, NatureConstr nature>
    Solution 
    vi_gs_cst(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    Solution vi_gs_l1_rob(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Robust Gauss-Seidel value iteration variant (not parallelized). The natures policy is
           constrained using L1 constraints and is worst-case.

           This function is suitable for computing the value function of a finite state MDP. If
           the states are ordered correctly, one iteration is enough to compute the optimal value function.
           Since the value function is updated from the first state to the last, the states should be ordered
           in reverse temporal order.

           Because this function updates the array value during the iteration, it may be
           difficult to parallelize.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_cst<SolutionType::Robust, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
    };

    Solution vi_gs_l1_opt(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Optimistic Gauss-Seidel value iteration variant (not parallelized). The natures policy is
           constrained using L1 constraints and is best-case.

           This function is suitable for computing the value function of a finite state MDP. If
           the states are ordered correctly, one iteration is enough to compute the optimal value function.
           Since the value function is updated from the first state to the last, the states should be ordered
           in reverse temporal order.

           Because this function updates the array value during the iteration, it may be
           difficult to parallelize.

           This is a generic version, which works for best/worst-case optimization and
           arbitrary constraints on nature (given by the function nature). Average case constrained
           nature is not supported.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
    };

    template<SolutionType type>
    Solution vi_jac_gen(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

    Solution vi_jac_rob(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Robust Jacobi value iteration variant. The nature behaves as worst-case.
           This method uses OpenMP to parallelize the computation.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */
         return vi_jac_gen<SolutionType::Robust>(valuefunction, discount, iterations, maxresidual);
    };

    Solution vi_jac_opt(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Optimistic Jacobi value iteration variant. The nature behaves as best-case.
           This method uses OpenMP to parallelize the computation.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */
         return vi_jac_gen<SolutionType::Optimistic>(valuefunction, discount, iterations, maxresidual);
    };

    Solution vi_jac_ave(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Average Jacobi value iteration variant. The nature behaves as average-case.
           This method uses OpenMP to parallelize the computation.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */
         return vi_jac_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
    };

    template<SolutionType type,NatureConstr nature>
    Solution vi_jac_cst(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;


    Solution vi_jac_l1_rob(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Robust Jacobi value iteration variant with constrained nature. The nature is constrained
           by an L1 norm.

           This method uses OpenMP to parallelize the computation.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

         return vi_jac_cst<SolutionType::Robust, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
    };

    Solution vi_jac_l1_opt(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Optimistic Jacobi value iteration variant with constrained nature. The nature is constrained
           by an L1 norm.

           This method uses OpenMP to parallelize the computation.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

         return vi_jac_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
    };


    // modified policy iteration
    template<SolutionType type>
    Solution mpi_jac_gen(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const;

    Solution mpi_jac_rob(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const{

        /**
           Robust modified policy iteration using Jacobi value iteration in the inner loop.
           The nature behaves as worst-case.

           This method generalizes modified policy iteration to robust MDPs.
           In the value iteration step, both the action *and* the outcome are fixed.

           Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

           \param valuefunction Initial value function
           \param discount Discount factor
           \param iterations_pi Maximal number of policy iteration steps
           \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
           \param iterations_vi Maximal number of inner loop value iterations
           \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                        This value should be smaller than maxresidual_pi

           \return Computed (approximate) solution
         */

         return mpi_jac_gen<SolutionType::Robust>(valuefunction, discount, iterations_pi, maxresidual_pi,
                     iterations_vi, maxresidual_vi);

    };

    Solution mpi_jac_opt(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const{

        /**
           Optimistic modified policy iteration using Jacobi value iteration in the inner loop.
           The nature behaves as best-case.

           This method generalizes modified policy iteration to robust MDPs.
           In the value iteration step, both the action *and* the outcome are fixed.

           Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

           \param valuefunction Initial value function
           \param discount Discount factor
           \param iterations_pi Maximal number of policy iteration steps
           \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
           \param iterations_vi Maximal number of inner loop value iterations
           \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                        This value should be smaller than maxresidual_pi

           \return Computed (approximate) solution
         */

         return mpi_jac_gen<SolutionType::Optimistic>(valuefunction, discount, iterations_pi, maxresidual_pi,
                     iterations_vi, maxresidual_vi);

    };

    Solution mpi_jac_ave(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const{

        /**
           Average modified policy iteration using Jacobi value iteration in the inner loop.
           The nature behaves as average-case.

           This method generalizes modified policy iteration to robust MDPs.
           In the value iteration step, both the action *and* the outcome are fixed.

           Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

           \param valuefunction Initial value function
           \param discount Discount factor
           \param iterations_pi Maximal number of policy iteration steps
           \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
           \param iterations_vi Maximal number of inner loop value iterations
           \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                        This value should be smaller than maxresidual_pi

           \return Computed (approximate) solution
         */

         return mpi_jac_gen<SolutionType::Average>(valuefunction, discount, iterations_pi, maxresidual_pi,
                     iterations_vi, maxresidual_vi);
    };


    template<SolutionType type, NatureConstr nature>
    Solution mpi_jac_cst(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const;

    Solution mpi_jac_l1_rob(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const{
        /**
           Robust modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
           The constraints are defined by the L1 norm and the nature is worst-case.

           This method generalized modified policy iteration to the robust MDP. In the value iteration step,
           both the action *and* the outcome are fixed.

           Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

           \param valuefunction Initial value function
           \param discount Discount factor
           \param iterations_pi Maximal number of policy iteration steps
           \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
           \param iterations_vi Maximal number of inner loop value iterations
           \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                        This value should be smaller than maxresidual_pi

           \return Computed (approximate) solution
         */

         return mpi_jac_cst<SolutionType::Robust,worstcase_l1>(valuefunction, discount, iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi);
    };

    Solution mpi_jac_l1_opt(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                     unsigned long iterations_vi, prec_t maxresidual_vi) const{
        /**
           Optimistic modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
           The constraints are defined by the L1 norm and the nature is best-case.

           This method generalized modified policy iteration to the robust MDP. In the value iteration step,
           both the action *and* the outcome are fixed.

           Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

           \param valuefunction Initial value function
           \param discount Discount factor
           \param iterations_pi Maximal number of policy iteration steps
           \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
           \param iterations_vi Maximal number of inner loop value iterations
           \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                        This value should be smaller than maxresidual_pi

           \return Computed (approximate) solution
         */

         return mpi_jac_cst<SolutionType::Optimistic,worstcase_l1>(valuefunction, discount, iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi);
    };

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
}

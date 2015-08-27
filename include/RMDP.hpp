#ifndef RMDP_H
#define RMDP_H

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
    Solution vi_gs_gen(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallelized). This is a generic function,
           which can compute any solution type (robust, optimistic, or average).

           Because this function updates the array value during the iteration, it may be
           difficult to paralelize.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        if(valuefunction.size() != states.size())
            throw invalid_argument("Incorrect dimensions of value function.");

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
                    newvalue = make_tuple(avgvalue.first,-1,avgvalue.second);
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

    Solution vi_gs_rob(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant (not parallelized). The outcomes are
           selected using worst-case nature.

           Because this function updates the array value during the iteration, it may be
           difficult to paralelize easily.

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

           Because this function updates the array value during the iteration, it may be
           difficult to paralelize easily.

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

           Because this function updates the array value during the iteration, it may be
           difficult to paralelize easily.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
    };

    template<SolutionType type, pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    Solution vi_gs_cst(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Gauss-Seidel value iteration variant with constrained nature(not parallelized).
           The natures policy is constrained, given by the function Nature.

           Because this function updates the array value during the iteration, it may be
           difficult to parallelize.

           This is a generic version, which works for best/worst-case optimization and
           arbitrary constraints on nature (given by the function Nature). Average case constrained
           nature is not supported.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */
        if(valuefunction.size() != this->states.size())
            throw invalid_argument("incorrect size of value function");

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
                    newvalue = state.max_min_cst<Nature>(valuefunction, discount);
                    break;
                case SolutionType::Optimistic:
                    newvalue = state.max_max_cst<Nature>(valuefunction, discount);
                    break;
                default:
                    static_assert(type != SolutionType::Robust || type != SolutionType::Optimistic, "Unknown/invalid (average not supported) optimization type.");
                    throw invalid_argument("Unknown/invalid (average not supported) optimization type.");
                }
                residual = max(residual, abs(valuefunction[s] - get<2>(newvalue) ));
                valuefunction[s] = get<2>(newvalue);
                outcome_dists[s] = get<1>(newvalue);
                policy[s] = get<0>(newvalue);
            }
        }
        return Solution(valuefunction,policy,outcome_dists,residual,i);
    };

    Solution vi_gs_l1_rob(vector<prec_t> valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Robust Gauss-Seidel value iteration variant (not parallelized). The natures policy is
           constrained using L1 constraints and is worst-case.

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

           Because this function updates the array value during the iteration, it may be
           difficult to parallelize.

           This is a generic version, which works for best/worst-case optimization and
           arbitrary constraints on nature (given by the function Nature). Average case constrained
           nature is not supported.

           \param valuefunction Initial value function. Passed by value, because it is modified.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        return vi_gs_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
    };

    template<SolutionType type>
    Solution vi_jac_gen(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Jacobi value iteration variant. The behavior of the nature depends on the values
           of parameter type. This method uses OpenMP to parallelize the computation.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        if(valuefunction.size() != states.size()){
            throw invalid_argument("incorrect dimension of the value function.");
        }

        vector<long> policy(states.size());
        vector<long> outcomes(states.size());

        vector<prec_t> oddvalue = valuefunction;        // set in even iterations (0 is even)
        vector<prec_t> evenvalue = valuefunction;       // set in odd iterations
        vector<prec_t> residuals(valuefunction.size());

        prec_t residual = numeric_limits<prec_t>::infinity();
        size_t i;

        for(i = 0; i < iterations && residual > maxresidual; i++){
            vector<prec_t> & sourcevalue = i % 2 == 0 ? oddvalue  : evenvalue;
            vector<prec_t> & targetvalue = i % 2 == 0 ? evenvalue : oddvalue;

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                const auto& state = states[s];

                pair<long,prec_t> avgvalue;
                tuple<long,long,prec_t> newvalue;

                switch(type){
                case SolutionType::Robust:
                    newvalue = state.max_min(sourcevalue,discount);
                    break;
                case SolutionType::Optimistic:
                    newvalue = state.max_max(sourcevalue,discount);
                    break;
                case SolutionType::Average:
                    avgvalue = state.max_average(sourcevalue,discount);
                    newvalue = make_tuple(avgvalue.first,-1,avgvalue.second);
                    break;
                default:
                    throw invalid_argument("Unknown optimization type.");
                }

                residuals[s] = abs(sourcevalue[s] - get<2>(newvalue));
                targetvalue[s] = get<2>(newvalue);

                policy[s] = get<0>(newvalue);
                outcomes[s] = get<1>(newvalue);
            }
            residual = *max_element(residuals.begin(),residuals.end());
        }
        vector<prec_t> & valuenew = i % 2 == 0 ? oddvalue : evenvalue;
        return Solution(valuenew,policy,outcomes,residual,i);
    };

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

    template<SolutionType type,pair<vector<prec_t>,prec_t> (*Nature)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t)>
    Solution vi_jac_cst(vector<prec_t> const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
        /**
           Jacobi value iteration variant with constrained nature. The outcomes are
           selected using Nature function.

           This method uses OpenMP to parallelize the computation.

           This is a generic version, which works for best/worst-case optimization and
           arbitrary constraints on nature (given by the function Nature). Average case constrained
           nature is not supported.

           \param valuefunction Initial value function.
           \param discount Discount factor.
           \param iterations Maximal number of iterations to run
           \param maxresidual Stop when the maximal residual falls below this value.
         */

        if(valuefunction.size() != this->states.size()){
            throw invalid_argument("incorrect dimension of the value function.");
        }

        vector<long> policy(this->states.size());
        vector<vector<prec_t>> outcome_dists(this->states.size());

        vector<prec_t> oddvalue = valuefunction;            // set in even iterations (0 is even)
        vector<prec_t> evenvalue = valuefunction;           // set in odd iterations
        vector<prec_t> residuals(valuefunction.size());

        prec_t residual = numeric_limits<prec_t>::infinity();
        size_t i;

        for(i = 0; i < iterations && residual > maxresidual; i++){
            vector<prec_t> & sourcevalue = i % 2 == 0 ? oddvalue  : evenvalue;
            vector<prec_t> & targetvalue = i % 2 == 0 ? evenvalue : oddvalue;

            #pragma omp parallel for
            for(auto s = 0l; s <  (long)this->states.size(); s++){
                const auto& state = this->states[s];

                tuple<long,vector<prec_t>,prec_t> newvalue;
                switch(type){
                case SolutionType::Robust:
                    newvalue = state.max_min_l1(sourcevalue,discount);
                    break;
                case SolutionType::Optimistic:
                    newvalue = state.max_max_l1(sourcevalue,discount);
                    break;
                default:
                    static_assert(type != SolutionType::Robust || type != SolutionType::Optimistic, "Unknown/invalid (average not supported) optimization type.");
                    throw invalid_argument("Unknown/invalid (average not supported) optimization type.");
                }

                residuals[s] = abs(sourcevalue[s] - get<2>(newvalue));
                targetvalue[s] = get<2>(newvalue);
                outcome_dists[s] = get<1>(newvalue);
                policy[s] = get<0>(newvalue);

            }
            residual = *max_element(residuals.begin(),residuals.end());
        }

        vector<prec_t> & valuenew = i % 2 == 0 ? oddvalue : evenvalue;

        return Solution(valuenew,policy,outcome_dists,residual,i);
    };


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
                     unsigned long iterations_vi, prec_t maxresidual_vi) const{

        /**
           Modified policy iteration using Jacobi value iteration in the inner loop.
            The template parameter type determines the behavior of nature.

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

        if(valuefunction.size() != this->states.size())
            throw invalid_argument("Incorrect size of value function.");


        vector<long> policy(this->states.size());
        vector<long> outcomes(this->states.size());

        vector<prec_t> oddvalue = valuefunction;        // set in even iterations (0 is even)
        vector<prec_t> evenvalue = valuefunction;       // set in odd iterations

        vector<prec_t> residuals(valuefunction.size());

        prec_t residual_pi = numeric_limits<prec_t>::infinity();

        size_t i; // defined here to be able to report the number of iterations

        vector<prec_t> * sourcevalue = & oddvalue;
        vector<prec_t> * targetvalue = & evenvalue;

        for(i = 0; i < iterations_pi; i++){

            std::swap<vector<prec_t>*>(targetvalue, sourcevalue);

            prec_t residual_vi = numeric_limits<prec_t>::infinity();

            // update policies
            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                const auto& state = states[s];

                pair<long,prec_t> avgvalue;
                tuple<long,long,prec_t> newvalue;

                // TODO: would removing the switch improve performance?
                switch(type){
                case SolutionType::Robust:
                    newvalue = state.max_min(*sourcevalue,discount);
                    break;
                case SolutionType::Optimistic:
                    newvalue = state.max_max(*sourcevalue,discount);
                    break;
                case SolutionType::Average:
                    avgvalue = state.max_average(*sourcevalue,discount);
                    newvalue = make_tuple(avgvalue.first,-1,avgvalue.second);
                    break;
                default:
                    throw invalid_argument("unknown optimization type.");
                }

                residuals[s] = abs((*sourcevalue)[s] - get<2>(newvalue));
                (*targetvalue)[s] = get<2>(newvalue);

                policy[s] = get<0>(newvalue);
                outcomes[s] = get<1>(newvalue);
            }

            residual_pi = *max_element(residuals.begin(),residuals.end());

            // the residual is sufficiently small
            if(residual_pi <= maxresidual_pi)
                break;

            // compute values using value iteration
            for(size_t j = 0; j < iterations_vi && residual_vi > maxresidual_vi; j++){

                swap(targetvalue, sourcevalue);

                #pragma omp parallel for
                for(auto s = 0l; s < (long) states.size(); s++){
                    prec_t newvalue;

                    switch(type){
                    case SolutionType::Robust:
                    case SolutionType::Optimistic:
                        newvalue = states[s].fixed_fixed(*sourcevalue,discount,policy[s],outcomes[s]);
                        break;
                    case SolutionType::Average:
                        newvalue = states[s].fixed_average(*sourcevalue,discount,policy[s]);
                        break;
                    default:
                        throw invalid_argument("Unknown optimization type.");
                    }

                    residuals[s] = abs((*sourcevalue)[s] - newvalue);
                    (*targetvalue)[s] = newvalue;
                }
                residual_vi = *max_element(residuals.begin(),residuals.end());
            }
        }

        vector<prec_t> & valuenew = *targetvalue;

        return Solution(valuenew,policy,outcomes,residual_pi,i);

    };

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

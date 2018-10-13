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

#include "craam/definitions.hpp"
#include "craam/optimization/optimization.hpp"
#include "craam/optimization/bisection.hpp"
#include "craam/Transition.hpp"
#include "craam/optimization/srect_gurobi.hpp"
#include <functional>

namespace craam {namespace algorithms {namespace nats{

// *******************************************************
// SA Nature definitions
// *******************************************************

/**
 * L1 robust response
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1{
protected:
    vector<numvec> budgets;
public:
    robust_l1(vector<numvec> budgets) : budgets(move(budgets)) {}

    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long stateid,long actionid,
                const numvec& nominalprob,const numvec& zfunction) const{
        assert(stateid > 0 && stateid < long(budgets.size()));
        assert(actionid > 0 && actionid < long(budgets[stateid].size()));

        return worstcase_l1(zfunction,nominalprob,budgets[stateid][actionid]);
    }
};

/**
 * L1 robust response with weights for states
 * The length of the state vector depends on the MDP type. For the plain
 * MDP it is the number of non-zero probabilities, a for an RMDP it is the
 * number of outcomes.
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1w{
protected:
    vector<numvec> budgets;
    /// The weights are optional, if empty then uniform weights are used.
    /// The elements are over states, actions, and then next state values
    vector<vector<numvec>> weights;

public:
    /**
     * @param budgets One value for each state and action
     * @param weights State weights used in the L1 norm. One set of vectors for each state and action.
     *                Use and empty vector to specify uniform weights.
     */
    robust_l1w(vector<numvec> budgets, vector<vector<numvec>> weights)
        : budgets(move(budgets)), weights(weights) {}

    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long stateid,long actionid,
                const numvec& nominalprob,const numvec& zfunction) const{
        assert(stateid > 0 && stateid < long(budgets.size()));
        assert(actionid > 0 && actionid < long(budgets[stateid].size()));
        assert(zfunction.size() == weights[stateid][actionid].size());

        return worstcase_l1_w(zfunction,nominalprob,weights[stateid][actionid],budgets[stateid][actionid]);
    }
};

/**
 * L1 robust response with a untiform budget/threshold
 *
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1u{
protected:
    prec_t budget;
public:
    robust_l1u(prec_t budget) : budget(move(budget)) {}

    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long,long,
                const numvec& nominalprob,const numvec& zfunction) const{

        return worstcase_l1(zfunction,nominalprob,budget);
    }
};

/**
 * L1 robust response
 *
 * @see rsolve_mpi, rsolve_vi
 */
class optimistic_l1{
protected:
    vector<numvec> budgets;
public:
    optimistic_l1(vector<numvec> budgets) : budgets(move(budgets)) {}

    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long stateid,long actionid,
                const numvec& nominalprob,const numvec& zfunction) const{
        assert(stateid > 0 && stateid < long(budgets.size()));
        assert(actionid > 0 && actionid < long(budgets[stateid].size()));
        assert(nominalprob.size() == zfunction.size());

        numvec minusv(zfunction.size());
        transform(begin(zfunction), end(zfunction), begin(minusv), negate<prec_t>());
        auto&& result = worstcase_l1(minusv,nominalprob,budgets[stateid][actionid]);
        return make_pair(result.first, -result.second);
    }
};

/**
 * L1 robust response with a untiform budget/threshold
 *
 * @see rsolve_mpi, rsolve_vi
 */
class optimistic_l1u{
protected:
    prec_t budget;
public:
    optimistic_l1u(prec_t budget) : budget(move(budget)) {};

    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long,long,
                const numvec& nominalprob,const numvec& zfunction) const{

        assert(nominalprob.size() == zfunction.size());

        numvec minusv(zfunction.size());
        transform(begin(zfunction), end(zfunction), begin(minusv), negate<prec_t>());
        auto&& result = worstcase_l1(minusv,nominalprob,budget);
        return make_pair(result.first, -result.second);
    }
};

/// Absolutely worst outcome
struct robust_unbounded{
    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long,long,
                const numvec&, const numvec& zfunction)const{

        //assert(v.size() == p.size());
        numvec dist(zfunction.size(),0.0);
        size_t index = size_t(min_element(begin(zfunction), end(zfunction)) - begin(zfunction));
        dist[index] = 1;
        return make_pair(dist,zfunction[index]);
    }
};

/// Absolutely best outcome
struct optimistic_unbounded{
    /**
     * @brief Implements SANature interface
     */
    pair<numvec, prec_t> operator() (long,long,
                const numvec&, const numvec& zfunction)const{

        //assert(v.size() == p.size());
        numvec dist(zfunction.size(),0.0);
        size_t index = size_t(max_element(begin(zfunction), end(zfunction)) - begin(zfunction));
        dist[index] = 1;
        return make_pair(dist,zfunction[index]);
    }
};

// --------------- GUROBI START --------------------------------------------------------
#ifdef GUROBI_USE

/**
 * L1 robust response using gurobi (slower!). Allows for weighted L1 norm.
 *
 * min_p p^T z
 * s.t. sum_i w_i |p_i - hat{p}_i| <= xi
 *
 */
class robust_l1w_gurobi{
protected:
    vector<numvec> budgets;
    /// The weights are optional, if empty then uniform weights are used.
    /// The elements are over states, actions, and then next state values
    vector<vector<numvec>> weights;
    shared_ptr<GRBEnv> env;
public:
    /**
     * Automatically constructs a gurobi environment object. Weights are uniform
     * when not provided
     * @param budgets Budgets, with a single value for each MDP state and action
     */
    robust_l1w_gurobi(vector<numvec> budgets) : budgets(move(budgets)), weights(0) {
        env = make_shared<GRBEnv>();
        // make sure it is run in a single thread so it can be parallelized
        env->set(GRB_IntParam_OutputFlag, 0);
        env->set(GRB_IntParam_Threads, 1);
    };

    /**
     * Automatically constructs a gurobi environment object. Weights are considered
     * to be uniform.
     * @param budgets Budgets, with a single value for each MDP state and action
     * @param weights State weights used in the L1 norm. One set of vectors for each state and action.
     *                Use and empty vector to specify uniform weights.
     */
    robust_l1w_gurobi(vector<numvec> budgets, vector<vector<numvec>> weights) :
            budgets(move(budgets)), weights(move(weights)) {

        env = make_shared<GRBEnv>();
        // make sure it is run in a single thread so it can be parallelized
        env->set(GRB_IntParam_OutputFlag, 0);
        env->set(GRB_IntParam_Threads, 1);
    };

    /**
     * @param budgets Budgets, with a single value for each MDP state and action
     * @param grbenv Gurobi environment that will be used. Should be single-threaded
     *                and probably disable printout. This environment is NOT
     *                thread-safe.
     */
    robust_l1w_gurobi(vector<numvec> budgets, vector<vector<numvec>> weights, const shared_ptr<GRBEnv>& grbenv) :
            budgets(move(budgets)), weights(move(weights)), env(grbenv) {};

    /**
     * Implements the SANature interface
     */
    pair<numvec, prec_t> operator() (long stateid, long actionid,
                const numvec& nominalprob, const numvec& zfunction) const{
        assert(stateid > 0 && stateid < long(budgets.size()));
        assert(actionid > 0 && actionid < long(budgets[stateid].size()));

        // check for whether weights are being used
        if(weights.empty()){
            return worstcase_l1_w_gurobi(*env, zfunction, nominalprob, numvec(0),
                                     budgets[stateid][actionid]);
        }
        else{
            return worstcase_l1_w_gurobi(*env, zfunction, nominalprob, weights[stateid][actionid],
                                     budgets[stateid][actionid]);
        }

    }
};

#endif
// --------------- GUROBI END --------------------------------------------------------


// *******************************************************
// S Nature definitions
// *******************************************************

/**
 * S-rectangular L1 constraint with a single budget for every state
 * and optional weights for each action for each state.
 *
 * The state-action weights are used as follows:
 *
 */
class robust_s_l1{

protected:
    numvec budgets;
    vector<numvec> weights_a;
public:

    robust_s_l1(numvec budgets) : budgets(move(budgets)), weights_a(0) {}

    robust_s_l1(numvec budgets, vector<numvec> weights_a) :
                budgets(move(budgets)), weights_a(move(weights_a)) {}

    /**
     * Implements SNature interface
     */
    tuple<numvec,vector<numvec>,prec_t> operator() (long stateid,
                    const vector<numvec>& nominalprobs,
                    const vector<numvec>& zvalues) const{

        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        prec_t outcome; numvec actiondist;
        numvec sa_budgets;

        // compute the distribution of actions and the optimal budgets
        if(!weights_a.empty()){
            tie(outcome, actiondist, sa_budgets) = solve_srect_bisection(zvalues, nominalprobs,
                                                        budgets[stateid], weights_a[stateid]);
        } else {
            tie(outcome, actiondist, sa_budgets) = solve_srect_bisection(zvalues, nominalprobs,
                                                        budgets[stateid]);
        }
        assert(actiondist.size() == zvalues.size());
        assert(sa_budgets.size() == actiondist.size());

        // compute actual worst-case responses for all actions
        // and aggregate them in a sparse transition probability
        vector<numvec> new_probability; new_probability.reserve(actiondist.size());
        for(size_t a = 0; a < nominalprobs.size(); a++){
            // skip the ones that have not transition probability
            if(actiondist[a] > EPSILON)
                new_probability.push_back(worstcase_l1(zvalues[a], nominalprobs[a], sa_budgets[a]).first);
            else
                new_probability.push_back(numvec(0));
        }
        return make_tuple(move(actiondist), move(new_probability), outcome);
    }
};

#ifdef GUROBI_USE

/**
 * S-rectangular L1 constraint with a single budget for every state
 * and optional weights for each action for each state.
 *
 * The state-action weights are used as follows:
 *
 * WARNING: does not compute the probaility distributions of policies and will therefore
 * fail when used in modified policy iteration
 */
class robust_s_l1_gurobi{

protected:
    // a budget for every state
    numvec budgets;
    shared_ptr<GRBEnv> env;
public:
    /**
     * Automatically constructs a gurobi environment object. Weights are uniform
     * when not provided
     * @param budgets Budgets, with a single value for each MDP state
     */
    robust_s_l1_gurobi(numvec budgets) : budgets(move(budgets)) {
        env = make_shared<GRBEnv>();
        // make sure it is run in a single thread so it can be parallelized
        env->set(GRB_IntParam_OutputFlag, 0);
        env->set(GRB_IntParam_Threads, 1);
    };

    /**
     * @param env Gurobi environment to use
     * @param budgets Budgets, with a single value for each MDP state
     */
    robust_s_l1_gurobi(const shared_ptr<GRBEnv>& env, numvec budgets) :
        budgets(move(budgets)), env(env) {};

    /**
     * Implements SNature interface
     */
    tuple<numvec,vector<numvec>,prec_t> operator() (long stateid,
                    const vector<numvec>& nominalprobs,
                    const vector<numvec>& zvalues) const{

        assert(stateid >= 0 && stateid < long(budgets.size()));
        assert(nominalprobs.size() == zvalues.size());

        prec_t outcome; numvec actiondist;

        // compute the distribution of actions and the optimal budgets

        tie(actiondist, outcome) = srect_solve_gurobi(*env,zvalues, nominalprobs,
                                                        budgets[stateid]);

        assert(actiondist.size() == zvalues.size());

        vector<numvec> new_probability(actiondist.size());
        return make_tuple(move(actiondist), move(new_probability), outcome);
    }
};

#endif

}}}

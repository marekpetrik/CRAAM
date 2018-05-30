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

#ifdef GUROBI_USE

#include "gurobi/gurobi_c++.h"
#include <memory>

namespace craam {

using namespace std;

/**
 * Solve the s-rectangular solution using gurobi linear solver
 *
 * max_{d in R^S} min_{p in R^{A*S}} sum_{a in A} d_a * z_a^T p_a
 * s.t. 1^T d = 1
 *      sum_{a in A} \| p_a - \bar{p}_a \|_{1,w_a} <= \kappa
 *      1^T p_a = 1
 *      p_a >= 0
 *      d >= 0
 *
 * The inner minimization problem can be reformulated as the following *linear* program.
 * The dual variables corresponding each constraint is noted in parentheses.
 *
 * min_{p, \theta in \theta in R^{|A| x |S|}} sum_{a in A} (d_a * z_a^T p_a)
 * s.t. 1^T p_a = 1                                     (dual: x_a)
 *      p_a - \bar{p}_a >= - \theta_a                   (dual: y^n_a)
 *      \bar{p}_a - p_a >= - \theta_a                   (dual: y^p_a)
 *      - sum_{a in A} w_a\tr \theta_a >= - \kappa      (dual: \lambda)
 *      p >= 0
 *
 * Dualizing the inner optimization problem, we get the full linear program for computing s-rectangular Bellman updates:
 * max_{d,x in R^{|A|},\lambda in R, y^p,y^n in R^{|S| x |A|}}
 *          sum_{a in A} ( x_a - \bar{p}_a\tr (y^n_a - y^p_a) ) - \kappa \lambda
 *  s.t.    1^T d = 1       d >= 0
 *          - y^p_a + y^n_a + x * 1         <= d_a z_a       a in A
 *          y^p_a + y^n_a - \lambda * w_a   <= 0             a in A
 *          y^p >= 0      y^n >= 0
 *          \lambda >= 0
 *
 * @param z Expected returns for each state and action (a state-length vector for each action)
 * @param pbar Nominal transition probability (a state-length vector for each action)
 * @param w Weights assigned to the L1 errors (optional). A uniform vector of all ones if omitted
 *
 * @returns A pair with: policy, objective value
 */
std::pair<numvec, double> srect_solve_gurobi(const GRBEnv& env, const numvecvec& z, const numvecvec& pbar, const prec_t kappa,
                                             const numvecvec& w = numvecvec(0)){

    // general constants values
    const double inf = std::numeric_limits<prec_t>::infinity();

    assert(pbar.size() == z.size());
    assert(w.empty() || w.size() == z.size());

    // helpful numbers of actions
    const size_t nactions = pbar.size();
    // number of transition states for each action
    std::vector<size_t> statecounts(nactions);
    transform(pbar.cbegin(), pbar.cend(), statecounts.begin(), [](const numvec& v){return v.size();});
    // the number of states per action does not need to be the same
    // (when transitions are sparse)
    const size_t nstateactions = accumulate(statecounts.cbegin(), statecounts.cend(), size_t(0));

    // construct the LP model
    GRBModel model = GRBModel(env);

    // Create varables: duals of the nature problem
    auto x = std::unique_ptr<GRBVar[]>(model.addVars(numvec(nactions,-inf).data(), nullptr,
                               nullptr,
                               std::vector<char>(nactions,GRB_CONTINUOUS).data(),
                                nullptr, int(nactions)));
    //  outer loop: actions, inner loop: next state
    auto yp = std::unique_ptr<GRBVar[]>(model.addVars(nullptr, nullptr,
                               nullptr,
                               std::vector<char>(nstateactions,GRB_CONTINUOUS).data(),
                               nullptr, int(nstateactions)));
    auto yn = std::unique_ptr<GRBVar[]>(model.addVars(nullptr, nullptr,
                               nullptr,
                               std::vector<char>(nstateactions,GRB_CONTINUOUS).data(),
                               nullptr, int(nstateactions)));

    auto lambda = model.addVar( 0, inf, -kappa, GRB_CONTINUOUS, "lambda");

    // primal variables for the nature
    auto d = std::unique_ptr<GRBVar[]>(model.addVars(numvec(nactions,0).data(), nullptr,
                               numvec(nactions,0).data(),
                               std::vector<char>(nactions,GRB_CONTINUOUS).data(),
                               nullptr, int(nactions)));
    // objective
    GRBLinExpr objective;

    size_t i = 0;
    // constraints dual to variables of the inner problem
    for(size_t actionid = 0; actionid < nactions; actionid++){
        objective += x[actionid];
        for(size_t stateid = 0; stateid < statecounts[actionid]; stateid++){
            // objective
            objective += - pbar[actionid][stateid] * yp[i];
            objective += pbar[actionid][stateid] * yn[i];
            // dual for p
            model.addConstr(x[actionid] - yp[i] + yn[i] <= d[actionid] * z[actionid][stateid]);
            // dual for z
            double weight = w.size() > 0 ? w[actionid][stateid] : 1.0;
            model.addConstr(-lambda * weight + yp[i] + yn[i] <= 0);
            // update the counter (an absolute index for each variable)
            i++;
        }
    }
    objective += - lambda * kappa;

    // constraint on the policy pi
    GRBLinExpr ones;
    ones.addTerms(numvec(nactions,1.0).data(),d.get(), int(nactions));
    model.addConstr(ones, GRB_EQUAL, 1);

    // set objective
    model.setObjective(objective, GRB_MAXIMIZE);

    // run optimization
    model.optimize();

    // retrieve policy values
    numvec policy(nactions);
    for(size_t i = 0; i < nactions; i++){
        policy[i] = d[i].get(GRB_DoubleAttr_X);
    }

    // retrieve the worst-case response values
    return make_pair(move(policy), model.get(GRB_DoubleAttr_ObjVal));
}


}
#endif

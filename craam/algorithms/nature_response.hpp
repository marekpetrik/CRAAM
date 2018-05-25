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

#include <functional>

namespace craam::algorithms::nats{

// *******************************************************
// Nature definitions
// *******************************************************


/// L1 robust response
/*inline vec_scal_t robust_l1(const numvec& v, const numvec& p, prec_t threshold){
    assert(v.size() == p.size());
    return worstcase_l1(v,p,threshold);
}*/

/**
 * L1 robust response
 *
 * @see rsolve_mpi, rsolve_vi
 */
class robust_l1{
protected:
    vector<numvec> budgets;
public:
    robust_l1(vector<numvec> budgets) : budgets(move(budgets)) {};

    pair<numvec, prec_t> operator() (long stateid,long actionid,
                const numvec& nominalprob,const numvec& zfunction) const{
        assert(stateid > 0 && stateid < budgets.size());
        assert(actionid > 0 && actionid < budgets[stateid].size());

        return worstcase_l1(zfunction,nominalprob,budgets[stateid][actionid]);
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
    robust_l1u(prec_t budget) : budget(move(budget)) {};

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
    optimistic_l1(vector<numvec> budgets) : budgets(move(budgets)) {};

    pair<numvec, prec_t> operator() (long stateid,long actionid,
                const numvec& nominalprob,const numvec& zfunction) const{
        assert(stateid > 0 && stateid < budgets.size());
        assert(actionid > 0 && actionid < budgets[stateid].size());
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
    pair<numvec, prec_t> operator() (long,long,
                const numvec&, const numvec& zfunction)const{

        //assert(v.size() == p.size());
        numvec dist(zfunction.size(),0.0);
        size_t index = size_t(max_element(begin(zfunction), end(zfunction)) - begin(zfunction));
        dist[index] = 1;
        return make_pair(dist,zfunction[index]);
    }
};


#ifdef GUROBI_USE
/// L1 robust response using gurobi (slower!)
inline vec_scal_t robust_l1_gurobi(const numvec& v, const numvec& p, pair<GRBEnv,prec_t> gur_budget){
    assert(v.size() == p.size());
    return worstcase_l1_w_gurobi(gur_budget.first,v,p,numvec(0), gur_budget.second);
}

inline vec_scal_t robust_l1_g(const numvec& v, const numvec& p, prec_t budget){
    static GRBEnv env = [](){
        GRBEnv env;     
        env.set(GRB_IntParam_OutputFlag, 0);
        // make sure it is run in a single thread so it can be parallelized
        env.set(GRB_IntParam_Threads, 1);
        return env;
        }();
    assert(v.size() == p.size());
    return worstcase_l1_w_gurobi(env,v,p,numvec(0), budget);
}
#endif

/// L1 robust response with weights for non-zero states
inline vec_scal_t robust_l1_w(const numvec& v, const numvec& p, pair<numvec, prec_t> weights_budget){
    auto [weights, budget] = weights_budget;
    assert(v.size() == p.size());
    assert(v.size() == weights.size());
    return worstcase_l1_w(v,p,weights,budget);
}

#ifdef GUROBI_USE
/// L1 robust response using gurobi (slower!)
inline vec_scal_t robust_l1_w_gurobi(const numvec& v, const numvec& p, tuple<GRBEnv,numvec,prec_t> gur_weights_budget){
    assert(v.size() == p.size());
    auto [grb, weights, budget] = gur_weights_budget;
    return worstcase_l1_w_gurobi(grb,v,p,weights,budget);
}
#endif


}

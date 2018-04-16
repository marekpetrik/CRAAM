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

namespace craam::algorithms{

// *******************************************************
// Nature definitions
// *******************************************************

/**
Function representing constraints on nature for s,a-rectangular nature.
The function computes the best response of nature and can be used in value iteration.

This function represents a nature which computes (in general) a randomized
policy (response). If the response is always deterministic, it may be better
to define and use a nature that computes and uses a deterministic response.

The parameters are the q-values v, the reference distribution p, and the threshold.
The function returns the worst-case solution and the objective value. The threshold can
be used to determine the desired robustness of the solution.

The value of threshold T may be a single number, or could also be a set of weights for
each state and action.

@see NatureResponseS
*/
template<class T>
using NatureResponse = pair<numvec, prec_t> (*)(numvec const& v, numvec const& p, T threshold);

/**
Represents an instance of nature and the corresponding threshold value
which can be used to directly compute the response.
*/
template<class T>
using NatureInstance = pair<NatureResponse<T>, T>;


/**
Function representing constraints on nature for s-rectangular nature.
The function computes the best response of nature and can be used in value iteration.

This function represents a nature which computes (in general) a randomized
policy (response). If the response is always deterministic, it may be better
to define and use a nature that computes and uses a deterministic response.

The parameters are the q-values v, the reference distribution p, and the threshold.
The function returns the worst-case solution and the objective value. The threshold can
be used to determine the desired robustness of the solution.

The value of threshold T may be a single number, or could also be a set of weights for
each state and action.

@see NatureResponse
*/
template<class T>
using NatureResponseS = tuple<numvec, numvec, prec_t> (*)(numvec const& v, numvec const& p, T threshold);

/**
Represents an instance of nature and the corresponding threshold value
which can be used to directly compute the response.
*/
template<class T>
using NatureInstanceS = pair<NatureResponseS<T>, T>;

/// L1 robust response
inline vec_scal_t robust_l1(const numvec& v, const numvec& p, prec_t threshold){
    assert(v.size() == p.size());
    return worstcase_l1(v,p,threshold);
}

#ifdef GUROBI_USE
/// L1 robust response using gurobi (slower!)
inline vec_scal_t robust_l1_gurobi(const numvec& v, const numvec& p, pair<GRBEnv,prec_t> gur_budget){
    assert(v.size() == p.size());
    return worstcase_l1_w_gurobi(gur_budget.first,v,p,numvec(0), gur_budget.second);
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


/// L1 optimistic response
inline vec_scal_t optimistic_l1(const numvec& v, const numvec& p, prec_t threshold){
    assert(v.size() == p.size());
    numvec minusv(v.size());
    transform(begin(v), end(v), begin(minusv), negate<prec_t>());
    auto&& result = worstcase_l1(minusv,p,threshold);
    return make_pair(result.first, -result.second);
}

/// worst outcome, threshold is ignored
template<class T>
inline vec_scal_t robust_unbounded(const numvec& v, const numvec&, T){
    assert(v.size() == p.size());
    numvec dist(v.size(),0.0);
    size_t index = size_t(min_element(begin(v), end(v)) - begin(v));
    dist[index] = 1;
    return make_pair(dist,v[index]);
}

/// best outcome, threshold is ignored
template<class T>
inline vec_scal_t optimistic_unbounded(const numvec& v, const numvec&, T){
    assert(v.size() == p.size());
    numvec dist(v.size(),0.0);
    size_t index = size_t(max_element(begin(v), end(v)) - begin(v));
    dist[index] = 1;
    return make_pair(dist,v[index]);
}


}

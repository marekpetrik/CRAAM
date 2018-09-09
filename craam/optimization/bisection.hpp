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
#include "optimization.hpp"

#include <tuple>


namespace craam {

using namespace std;


/**
    Computes a value of a piecewise linear function h(x)

    The lower bound of the range is closed and a smaller parameter values than the lower limit is not allowed.

    The upper bound of the range is open and the function is assumed to be constant
    going to the infinity.

    @param knots Knots of the function (in parameter x). The array must be sorted increasingly.
    @param values Values in the knots (h(k) for knot k)
    @param x The parameter value
    @return Value of the piecewise linear function and the index of the knot. The value is
                between knots[index-1] and knots[index]. If the parameter is past the
                largest knot, then index points beyond the end of the array
 */
std::pair<prec_t, size_t> piecewise_linear(const numvec& knots, const numvec& values, prec_t x){

    const prec_t epsilon = 1e-10;

    assert(knots.size() == values.size());
    assert(is_sorted(knots.cbegin(), knots.cend()));

    // first element that is greater than or equal to x
    size_t index = size_t(distance(knots.cbegin(),  lower_bound(knots.cbegin(), knots.cend(), x)));

    // check for function boundaries
    if(index <= 0){
        if(x < knots.front() - epsilon)
            throw std::invalid_argument("Parameter x is smaller than the valid range.");
        else
            return {values.front(), index};
    }
    // if all elements are smaller than the last element is returned, so just return the last value
    if(index >= knots.size() - 1 && x > knots.back()){
        return {values.back(), index+1};
    }

    // the linear segment is between (index - 1) and the (index), so we need to average them
    prec_t x0 = knots[index-1];
    prec_t x1 = knots[index];
    // x = alpha * x0 + (1 - alpha) * x1
    // alpha = (x - x1) / (x0 - x1)
    prec_t alpha  = (x1 - x) / (x1 - x0);
    assert(alpha >= 0 && alpha <= 1);

    prec_t value = alpha * values[index-1] + (1-alpha) * values[index];
    return {value, size_t(index)};
}


/**
@brief solve_srect_bisection Computes the optimal objective value of the s-rectangular problem

Solves the optimization problem:

max_d min_{xi,p} sum_a d(a) p_a^T z_a
s.t.    1^T pi = 1, pi >= 0
        sum_a xi(a) wa(a) <= psi
        || p_a - pbar_a ||_{1,ws_a} <= xi_a

The algorithm works by reformulating the problem to:

min_u {u : sum_a xi(a) wa(a) <= psi, q_a^{-1}(xi_a) <= u}, where
d l
q_a^{-1}(u_a) = min_{p,t} || p - pbar ||_{1,ws_a}
s.t.    z^T e <= b
        1^T e = 1
        p >= 0

The function q_a^{-1} is represented by a piecewise linear function.

@note Note that the returned xi values may sum to less than psi. This happens when an
      an action is not active and xi for the particular action is already at its
      maximal value.


@param z Rewards (or values) for all actions
@param p Nominal distributions for all actions
@param psi Bound on the sum of L1 deviations
@param wa Optional set of weights on action errors
@param ws Optional set of weights on staet errors (using these values can significantly slow the computation)
@param gradients Optional structure that holds pre-computed gradients to speed up the computation
                     of the weighted L1 response. Only used with weighted L1 computation; the
                     unweighted L1 is too fast to make this useful.

@return Objective value, policy (d),
        nature's deviation from nominal probability distribution (xi)
*/
tuple<prec_t,numvec,numvec>
solve_srect_bisection(const vector<numvec>& z, const vector<numvec>& pbar, const prec_t psi,
                        const numvec& wa = numvec(0), const vector<numvec> ws = vector<numvec>(0),
                        const vector<GradientsL1_w> gradients = vector<GradientsL1_w>(0)){

    // make sure that the inputs make sense
    if(z.size() != pbar.size()) throw invalid_argument("pbar and z must have the same size.");
    if(psi < 0.0) throw invalid_argument("psi must be non-negative");
    if(!wa.empty() && wa.size() != z.size()) throw invalid_argument("wa must be the same size as pbar and z.");
    if(!ws.empty() && ws.size() != z.size()) throw invalid_argument("ws must be the same size as pbar and z.");
    if(!gradients.empty() && gradients.size() != z.size()) throw invalid_argument("gradients must be the same length as pbar and z.");

    // define the number of actions
    const size_t nactions = z.size();

    if(nactions == 0) throw invalid_argument("cannot be called with 0 actions");

    for(size_t a = 0; a < nactions; a++){
        assert(abs(1.0 - accumulate(pbar[a].cbegin(), pbar[a].cend(), 0.0) ) < EPSILON);
        assert(*min_element(pbar[a].cbegin(), pbar[a].cend()) >= 0.0);
    }

    // define the knots and the corresponding values for the piecewise linear q_a^{-1}
    vector<numvec>  knots(nactions),        // knots are the possible values of q_a^{-1}
                    values(nactions);       // corresponding values of xi_a for the corresponsing value of g_a

    // minimal and maximal possible values of u
    prec_t min_u = -numeric_limits<prec_t>::infinity(),
           max_u = -numeric_limits<prec_t>::infinity();

    for(size_t a = 0; a < nactions; a++){
        // compute the piecewise linear approximation
        assert(z[a].size() == pbar[a].size());

        // check whether state weights are being used,
        // this determines which knots function would be called
        if(ws.empty()){
            tie(knots[a], values[a]) = worstcase_l1_knots(z[a], pbar[a]);
        }
        else{
            if(gradients.empty())
                tie(knots[a], values[a]) = worstcase_l1_w_knots(z[a], pbar[a], ws[a]);
            else
                tie(knots[a], values[a]) = worstcase_l1_w_knots(gradients[a], z[a], pbar[a], ws[a]);
        }

        // knots are in the reverse order than what we want here
        // This step could be eliminated by changing the function worstcase_l1_knots to generate the values in a reverse order
        reverse(knots[a].begin(), knots[a].end());
        reverse(values[a].begin(), values[a].end());

        // make sure that the largest knot has xi = 0
        assert(abs(values[a].back()) <= 1e-6);

        // update the lower and upper limits on u

        #ifdef __cpp_structured_bindings
        auto [minval, maxval] = minmax_element(knots[a].cbegin(), knots[a].cend());
        #else
        auto minmaxval = minmax_element(knots[a].cbegin(), knots[a].cend());
        auto minval = minmaxval.first;
        auto maxval = minmaxval.second;
        #endif
        //cout << "minval " << *minval << "  maxval " << *maxval << endl;
        // the function is infinite for values smaller than the minumum for any action
        min_u = max(*minval, min_u);
        max_u = max(*maxval, max_u);
    }

    // *** run a bisection search on the value of u. Treats u as a continuous variable, but
    //      identifies when the function becomes linear and then terminates with the precise
    //      solution
    // lower and upper bounds on the value of u.
    //  => u_lower: largest known value for which the problem is infeasible
    //  => u_upper: smallest known value for which the problem is feasible
    //  => u_pivot: the next value of u that should be examined
    prec_t u_lower = min_u, // feasible for this value of u, but that why we set the pivot there
           u_upper = max_u,
           u_pivot = (max_u + min_u) / 2;

    //cout << "min_u " << min_u << endl;
    //cout << "max_u " << max_u << endl;

    //indexes of the largest lower bound knots for u_lower and u_upper, and the pivot.
    // these are used to determine when the lower and upper bounds are on a single line
    sizvec indices_upper(nactions),
           indices_lower(nactions);

    // define vectors with the problem solutions
    numvec pi(nactions, 0);
    numvec xi(nactions);

    // sum of xi, to compute the final optimal value which is a linear combination of both
    prec_t xisum_lower = 0,
           xisum_upper = 0;

    // compute xisum_upper and indices,
    for(size_t a = 0; a < nactions; a++){    
        #ifdef __cpp_structured_bindings
        auto [xia, index] = piecewise_linear(knots[a], values[a], u_upper);
        #else
        double xia; size_t index;
        tie(xia, index) = piecewise_linear(knots[a], values[a], u_upper);
        #endif
        xisum_upper += xia * (wa.empty() ? 1.0 : wa[a]);
        indices_upper[a] = index;
    }

    // compute xisum_lower
    for(size_t a = 0; a < nactions; a++){
        #ifdef __cpp_structured_bindings
        auto [xia, index] = piecewise_linear(knots[a], values[a], u_lower);
        #else
        double xia; size_t index;
        tie(xia, index) = piecewise_linear(knots[a], values[a], u_lower);
        #endif
        assert((wa.empty() ? 1.0 : wa[a]) > 0.0);
        xisum_lower += xia * (wa.empty() ? 1.0 : wa[a]);
        indices_lower[a] = index;
        xi[a] = xia; // cache the solution in case the value is feasible
    }

    // need to handle the case when u_lower is feasible. Because the rest of the code
    // assumes that u_lower is infeasible
    if(xisum_lower <= psi){
        // index of the state which the index is 0
        size_t zero_index = size_t(distance(indices_lower.cbegin(),
                                            min_element(indices_lower.cbegin(), indices_lower.cend())));
        assert(indices_lower[zero_index] == 0);
        // the policy will be to take the action in which the index is 0
        // because the other actions will be worse; the derivative of
        // this action is infty
        pi[zero_index] = 1;

        // just return the solution value
        return make_tuple(u_lower, move(pi), move(xi));
    }

    // run the iteration until the piecewise until upper bounds and lower bounds are close
    while(u_upper - u_lower >= 1e-10){
        assert(u_lower <= u_upper);
        assert(u_pivot >= u_lower && u_pivot <= u_upper);

        // add up the values of xi and indices
        prec_t xisum = 0;
        sizvec indices_pivot(nactions);
        for(size_t a = 0; a < nactions; a++){
            #ifdef __cpp_structured_bindings
            auto [xia, index] = piecewise_linear(knots[a], values[a], u_pivot);
            #else
            double xia; size_t index;
            tie(xia, index) = piecewise_linear(knots[a], values[a], u_pivot);
            #endif
            xisum += xia * (wa.empty() ? 1.0 : wa[a]);
            indices_pivot[a] = index;
        }

        // set lower an upper bounds depending on whether the solution is feasible
        if(xisum <= psi){
            // solution is feasible
            u_upper = u_pivot;
            xisum_upper = xisum;
            indices_upper = move(indices_pivot);
        }else{
            // solution is infeasible
            u_lower = u_pivot;
            xisum_lower = xisum;
            indices_lower = move(indices_pivot);
        }
        // update the new pivot in the middle
        u_pivot = (u_lower + u_upper) / 2;

        // xisums decrease with u, make sure that this is indeed the case
        assert(xisum_lower >= xisum_upper);

        // if this is the same linear segment, then we can terminate and compute the
        // final solutions as a linear combination
        if(indices_lower == indices_upper){
            break;
        }
    }
    //cout << "xisum lower " << xisum_lower << "  xisum_upper " << xisum_upper << endl;
    //cout << "u_lower " << u_lower << "  u upper " << u_upper << endl;

    // compute the value as a linear combination of the individual values
    // alpha xisum_lower + (1-alpha) xisum_upper = psi
    // alpha = (psi - xisum_upper) / (xisum_lower - xisum_upper)
    prec_t alpha = (psi - xisum_upper) / (xisum_lower - xisum_upper);

    assert(alpha >= 0 && alpha <= 1);

    // the result is then: alpha * u_lower + (1-alpha) * u_upper
    prec_t u_result = alpha * u_lower + (1-alpha) * u_upper;

    // ***** NOW compute the primal solution (pi) ***************
    // this is based on computing pi such that the subderivative of
    // d/dxi ( sum_a pi_a f_a(xi_a) - lambda (sum_a xi_a f(a) ) ) = 0 for xi^*
    // and ignore the inactive actions (for which u is higher than the last segment)
    for(size_t a = 0; a < nactions; a++){
        #ifdef __cpp_structured_bindings
        auto [xia, index] = piecewise_linear(knots[a], values[a], u_result);
        #else
        double xia; size_t index;
        tie(xia, index) = piecewise_linear(knots[a], values[a], u_result);
        #endif
        xi[a] = xia;

        //cout << " index " << index << "/" << knots[a].size() << endl;
        assert(knots[a].size() == values[a].size());

        if(index == 0){
            cout << "z[a] = " << z[a] << endl;
            cout << "pbar[a] = " << pbar[a] << endl;
            cout << "knots = " << knots[a] << endl;
            cout << "values = " << values[a] << endl;

            // TODO: Can this ever happen?

            throw runtime_error("This should not happen (can happen when z's are all the same); index = 0 should be handled by the special case with u_lower feasible. u_lower = " +
                                                to_string(u_lower) + ", u_upper" + to_string(u_upper) + ", xisum_lower =" + to_string(xisum_lower) +
                                                ", xisum_upper = " + to_string(xisum_upper) +
                                                ", psi = " + to_string(psi) + ", knots = " + to_string(knots[a].size()));
        }

        // the value u lies between index - 1 and index
        // when it is outside of the largest knot, the derivative is 0 and policy will be 0 (this is when f_a(xi_a^*) < u^*)
        //      that case can be ignored because pi is already intialized
        if(index < knots[a].size()){
            // we are not outside of the largest knot or the smallest knot

            // compute the derivative of f (1/derivative of g)
            // prec_t derivative = (knots[a][index] - knots[a][index-1]) / (values[a][index] - values[a][index-1]);
            // pi[a] = 1/derivative;
            pi[a] = - (values[a][index] - values[a][index-1]) / (knots[a][index] - knots[a][index-1]);
        }

        //cout << "pi[a] " << pi[a] << endl;
        assert(pi[a] >= 0);
    }
    // normalize the probabilities
    prec_t pisum = accumulate(pi.cbegin(), pi.cend(), 0.0);
    // u_upper is chosen to be the upper bound, and thus at least one index should be within the range
    assert(pisum > 1e-5);
    // divide by the sum to normalize
    transform(pi.cbegin(), pi.cend(), pi.begin(), [pisum](prec_t t){return t/pisum;});

    return make_tuple(u_result, move(pi), move(xi));
}


}

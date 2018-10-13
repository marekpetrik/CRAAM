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

#include <tuple>
// if available, use gurobi
#ifdef GUROBI_USE
#include "gurobi/gurobi_c++.h"
#include <memory>   // unique_pointer for gurobi
#include <cmath>    // pow in gurobi
#endif


// The file includes methods for fast solutions of various optimization problems

namespace craam {

/**
@brief Worstcase distribution with a bounded deviation.

Efficiently computes the solution of:
min_p   p^T * z
s.t.    ||p - pbar|| <= xi
        1^T p = 1
        p >= 0

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the provide probability distribution sums to 1.

@see worstcase_l1_penalty
@param z Reward values
@param pbar Nominal probability distribution
@param t Bound on the L1 norm deviation
@return Optimal solution p and the objective value
*/
pair<numvec,double> worstcase_l1(numvec const& z, numvec const& pbar, prec_t xi){
    assert(*min_element(pbar.cbegin(), pbar.cend()) >= - THRESHOLD);
    assert(*max_element(pbar.cbegin(), pbar.cend()) <= 1 + THRESHOLD);
    assert(xi >= 0.0);
    assert(z.size() > 0 && z.size() == pbar.size());

    xi = clamp(xi, 0.0, 2.0);

    const size_t sz = z.size();
    // sort z values
    const vector<size_t> sorted_ind = sort_indexes<prec_t>(z);
    // initialize output probability distribution; copy the values because most may be unchanged
    numvec o(pbar);
    // pointer to the smallest (worst case) element
    size_t k = sorted_ind[0];
    // determine how much deviation is actually possible given the provided distribution
    prec_t epsilon = min(xi/2, 1-pbar[k]);
    // add all the possible weight to the smallest element (structure of the optimal solution)
    o[k] += epsilon;
    // start from the last element
    size_t i = sz - 1;
    // find the upper quantile that corresponds to the epsilon
    while(epsilon > 0){
        k = sorted_ind[i];
        // compute how much of epsilon remains and can be addressed by the current element
        auto diff = min( epsilon, o[k] );
        // adjust the output and epsilon accordingly
        o[k] -= diff;
        epsilon -= diff;
        i--;
    }
    prec_t r = inner_product(o.cbegin(),o.cend(),z.cbegin(), prec_t(0.0));
    return make_pair(move(o),r);
}

/**
@brief Worstcase deviation given a linear constraint. Used to compute s-rectangular solutions

Efficiently computes the solution of:
min_{p,t} ||p - q||
s.t.    z^T p <= b
        1^T p = 1
        p >= 0

When the problem is infeasible, then the returned objective value is infinite
and the solution is a vector of length 0.

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the provided probability distribution sums to 1.

@see worstcase_l1
@param z Reward values
@param p Distribution
@param b Constant in the linear inequality
@return Optimal solution p and the objective value (t).
        Important: returns the objective value and not the dot product
*/
pair<numvec,prec_t> worstcase_l1_deviation(numvec const& z, numvec const& p, prec_t b){
    assert(*min_element(p.cbegin(), p.cend()) >= - THRESHOLD);
    assert(*max_element(p.cbegin(), p.cend()) <= 1 + THRESHOLD);
    assert(b >= 0.0);
    assert(z.size() > 0 && z.size() == p.size());

    const size_t sz = z.size();
    // sort z values (increasing order)
    const vector<size_t> sorted_ind = sort_indexes<prec_t>(z);
    // initialize output probability distribution; copy the values because most may be unchanged
    numvec o(p);
    // initialize the difference t for the output (this is 1/2 of the output value)
    prec_t t = 0;
    // start with t = 0 and increase it progressively until the constraint is satisifed
    // epsilon is the remainder of the constraint that needs to be satisfied
    prec_t epsilon = inner_product(z.cbegin(), z.cend(), p.cbegin(), 0.0) - b;
    // now, simply add violation until the constraint is tight
    // start with the largest element and move towards the beginning
    size_t i = sz - 1;
    // cache the smallest element
    const prec_t smallest_z = z[sorted_ind[0]];
    while(epsilon > 0 && i > 0){
        size_t k = sorted_ind[i];
        // adjustment size
        prec_t derivative = z[k] - smallest_z;
        // compute how much of epsilon remains and can be addressed by the current element
        prec_t diff = min( epsilon / derivative, o[k]);
        // adjust the output and epsilon accordingly
        o[k] -= diff; t += diff;
        epsilon -= derivative*diff;
        i--;
    }
    // if there is still some value epsilon, then the solution is not feasible
    if(epsilon > 0){
        return make_pair(numvec(0), numeric_limits<prec_t>::infinity());
    }
    else{
        // adjust the smallest element
        o[sorted_ind[0]] += t;
        // the l1 norm is twice the difference for the smallest element
        return make_pair(move(o), 2 * t);
    }
}


/**
Identifies knots of the piecewise linear function of the worstcase l1-constrained
response.

Consider the function:
q^-1(b) = min_{p,t} ||p - q||_1
s.t.    z^T p <= b
        1^T p = 1
        p >= 0

The function returns the points of nonlinearity of q^{-1}(b).

The function is convex. It is infty as b -> -infty and constant as b -> infty.

@param z Reward values
@param p Nominal distribution.
@param presorted_ind (optional) Presorted indices for d. If length is 0 then the indexes are computed.

@return A pair: (knots = b values, function values = xi values)
        knots: Set of values xi for which the function is nonlinear. It is linear everywhere in between (where defined).
        The values are generated in a decreasing order.
        value: Corresponding values of t for the values above.
*/
pair<numvec, numvec> worstcase_l1_knots(const numvec& z, const numvec& p,
                                        const sizvec& presorted_ind = sizvec(0)){
    assert(z.size() == p.size());

    // sorts indexes if they are not provided
    sizvec sorted_cache(0);
    if(presorted_ind.empty()){
        sorted_cache = sort_indexes(z);
    }
    // sort z values (increasing order)
    const sizvec& sorted_ind = presorted_ind.empty() ? sorted_cache : presorted_ind;

    // cache the smallest element
    const prec_t smallest_z = z[sorted_ind.front()];

    // knots = values of b
    numvec knots; knots.reserve(z.size());
    // objective values = values of t
    numvec values; values.reserve(z.size());

    prec_t knot = inner_product(z.cbegin(), z.cend(), p.cbegin(), 0.0);
    prec_t value = 0;   // start with no value
    knots.push_back(knot);
    values.push_back(value);

    for(long k = long(z.size()) - 1; k > 0; k--){
        knot -= (z[sorted_ind[size_t(k)]] - smallest_z) * p[sorted_ind[size_t(k)]];
        value += 2*p[sorted_ind[size_t(k)]];
        knots.push_back(knot);
        values.push_back(value);
    }
    return make_pair(move(knots), move(values));
}

/**
 * Holds and computes the gradients for the homotopy methods. This is used by
 *          worstcase_l1_w and worstcase_l1_w_knots
 *
 * The function computes and sorts the possible basic feasible solutions of:
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{1,w} <= xi
 *
 */
class GradientsL1_w {
protected:
    numvec derivatives;         // derivate for each potential basic solution
    indvec donors;              // the index of the donor for each potential basic solution
    indvec receivers;           // the index of the receiver for each potential basic solution
    vector<bool> donor_greater; // whether the donor is greater than the nominal solution for each potential basic solution
    vector<size_t> sorted;      // order of elements after sorted increasingly according to the derivatives

public:
    /**
     * Constructs an empty structure
     */
    GradientsL1_w(){};

    /**
     * Computes the possible gradients and sorts them increasingly
     * @param z Objective function
     * @param w Weights in the definition of the L1 norm
     */
    GradientsL1_w(const numvec& z, const numvec& w){
        const double epsilon = 1e-10;
        size_t element_count = z.size();

        assert(z.size() == element_count);
        assert(w.size() == element_count);

        derivatives.reserve(element_count);
        donors.reserve(element_count+1);
        receivers.reserve(element_count+1);
        donor_greater.reserve(element_count+1);   // whether the donor p is greater than the corresponding pbar

        // identify possible receivers (must be less weight than all the smaller elements)
        vector<size_t> possible_receivers;
        { // limit the visibility of these variables
            vector<size_t> z_increasing = sort_indexes(z);
            double smallest_w = numeric_limits<double>::infinity();

            for(size_t iz : z_increasing){
                if(w[iz] < smallest_w){
                    possible_receivers.push_back(iz);
                    smallest_w = w[iz];
                }
            }
        }

        // ** compute derivatives for possible donor-receiver pairs

        // case a: donor is less or equal to pbar
        // donor
        for(size_t i = 0; i < element_count; i++){
            // receiver
            for(size_t j : possible_receivers){
                // cannot donate from a smaller value to a larger one; just skip it
                if(z[i] <= z[j]) continue;

                // case a: donor is less or equal to pbar
                derivatives.push_back( (- z[i] + z[j]) / (w[i] + w[j]) );
                donors.push_back(long(i));
                receivers.push_back(long(j));
                donor_greater.push_back(false);
            }
        }

        // case b: current donor value is greater than pbar and the weight change is non-negative (otherwise a contradiction with optimality)
        // donor (only possible receiver can be a donor here)
        for(size_t i : possible_receivers){
            // receiver
            for(size_t j : possible_receivers){
                // cannot donate from a smaller value to a larger one; just skip it
                if(z[i] <= z[j]) continue;

                if(abs(w[i]-w[j]) > epsilon && w[i] < w[j]){
                    // HACK!: adding the epsilon here makes sure that these basic solutions
                    // are preferred in case of ties. This is to prevent skipping
                    // over this kind of basis when it is tied with type a
                    derivatives.push_back( epsilon + (-z[i] + z[j]) / (-w[i] + w[j]) );
                    donors.push_back(long(i));
                    receivers.push_back(long(j));
                    donor_greater.push_back(true);
                }
            }
        }

        assert(donors.size() == receivers.size());
        assert(donor_greater.size() == receivers.size());

        sorted = sort_indexes(derivatives);
    }

    /** Returns the number of potential basic solutions generated */
    size_t size() const{ return derivatives.size();}

    /**
     * Returns parameters for the basic solution with gradients that
     * increase with increasing indexes
     * @param index Position, 0 is the smallest derivative, size() -1 is the largest one
     * @return (gradient, donor index, receiver index, does donor probability must be greater than nominal?)
     */
    tuple<double,size_t,size_t,bool> steepest_solution(size_t index) const{
        size_t e = sorted[index];
        return make_tuple(derivatives[e],donors[e],receivers[e],donor_greater[e]);
    }
};

/**
 * Solve the worst case response problem using a homotopy method.
 *
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{1,w} <= xi
 *
 * @param gradients Pre-computed greadients (n^2 worst-case complexity)
 * @param z Objective
 * @param pbar Nominal distribution
 * @param w Weights in the norm
 * @return the optimal solution and the objective value
 */
pair<numvec, double> worstcase_l1_w(const GradientsL1_w& gradients, const numvec& z, const numvec& pbar, const numvec& w, double xi){

    assert(pbar.size() == z.size());
    assert(*min_element(pbar.cbegin(), pbar.cend()) >= 0);
    assert(abs(accumulate(pbar.cbegin(), pbar.cend(), 0.0) - 1.0) < 1e-6);

    const double epsilon = 1e-10;

    // the working value of the new probability distribution
    numvec p = pbar;
    // remaining value of xi that needs to be allocated
    double xi_rest = xi;



    for(size_t k = 0; k < gradients.size(); k++){
        // edge index
        #ifdef __cpp_structured_bindings
        auto [ignore, donor, receiver, donor_greater] = gradients.steepest_solution(k);
        #else
        size_t donor, receiver; bool donor_greater; tie(std::ignore, donor, receiver, donor_greater) = gradients.steepest_solution(k);
        #endif

        // this basic solution is not applicable here, just skip it
        if(donor_greater && p[donor] <= pbar[donor]) continue;

        // No obvious reason how this could happen, but lets just make sure to flag it
        // this could happen because of ties, see the hack above
        if(!donor_greater && p[donor] > pbar[donor] + epsilon) {throw runtime_error("internal program error, unexpected value of p");}

        // make sure that the donor can give
        if(p[donor] < epsilon) continue;

        double weight_change = donor_greater ? (-w[donor] + w[receiver]) : (w[donor] + w[receiver]);
        assert(weight_change > 0);

        double donor_step = min(xi_rest / weight_change, (p[donor] > pbar[donor] + epsilon) ? (p[donor] - pbar[donor]) : p[donor]);
        p[donor] -= donor_step;
        p[receiver] += donor_step;
        xi_rest-= donor_step*weight_change;

        // stop if there is nothing left
        if(xi_rest < epsilon) break;
    }

    double objective = inner_product(p.cbegin(), p.cend(), z.cbegin(), 0.0);
    return make_pair(move(p),objective);
}

/**
 * @brief See the documentation for the overloaded function
 */
pair<numvec,double> worstcase_l1_w(const numvec& z, const numvec& pbar, const numvec& w, double xi){
    return worstcase_l1_w(GradientsL1_w(z,w),z,pbar,w,xi);
}

/**
Identifies knots of the piecewise linear function of the weighted worstcase l1-constrained
response.

Consider the function:
q^-1(u) = min_{p,t} ||p - pbar||_{1,w}
s.t.    z^T p <= u
        1^T p = 1
        p >= 0

The function returns the points of nonlinearity of q^{-1}(u). It probably works even when
p sums to less than 1.

The function is convex and non-increasing. It is infty as u -> -infty and constant as u -> infty.

@param gradients Pre-computed greadients (n^2 worst-case complexity)
@param z Reward values
@param pbar Nominal probability distribution.
@param w Weights used in the L1 norm

@return A pair: (knots = b values, function values = xi values)
        knots: Set of values xi for which the function is nonlinear. It is linear everywhere in between (where defined).
        The values are generated in a decreasing order.
        value: Corresponding values of t for the values above.
*/
inline pair<numvec, numvec>
worstcase_l1_w_knots(const GradientsL1_w& gradients, const numvec& z, const numvec& pbar, const numvec& w){

    const double epsilon = 1e-10;

    // the working value of the new probability distribution
    numvec p = pbar;

    // will hold the values and knots for the solution path
    numvec knots;
    numvec values;

    // initial value
    knots.push_back(inner_product(pbar.cbegin(), pbar.cend(), z.cbegin(), 0.0)); // u
    values.push_back(0.0);       // || ||_{1,w}

    // trace the value of the norm and update the norm difference as well as the value of the return (u)
    for(size_t k = 0; k < gradients.size(); k++){

        #ifdef __cpp_structured_bindings
        auto [ignore, donor, receiver, donor_greater] = gradients.steepest_solution(k);
        #else
        size_t donor, receiver; bool donor_greater; tie(std::ignore, donor, receiver, donor_greater) = gradients.steepest_solution(k);
        #endif

        // this basic solution is not applicable here, just skip it
        if(donor_greater && p[donor] <= pbar[donor]) continue;

        // No obvious reason how this could happen, but lets just make sure to flag it
        // this could happen because of ties, see the hack above
        if(!donor_greater && p[donor] > pbar[donor] + epsilon) {throw runtime_error("internal program error, unexpected value of p");}

        // make sure that the donor can give
        if(p[donor] < epsilon) continue;

        double weight_change = donor_greater ? (-w[donor] + w[receiver]) : (w[donor] + w[receiver]);
        assert(weight_change > 0);

        double donor_step = donor_greater ? (p[donor] - pbar[donor]) : p[donor];
        p[donor] -= donor_step;
        p[receiver] += donor_step;

        knots.push_back(knots.back() + donor_step * (z[receiver] - z[donor]) );
        values.push_back(values.back() + donor_step * weight_change );
    }

    return make_pair(move(knots), move(values));
}

/// See the overloaded method
pair<numvec, numvec> worstcase_l1_w_knots(const numvec& z, const numvec& pbar, const numvec& w){
    return worstcase_l1_w_knots(GradientsL1_w(z,w),z,pbar,w);
}


#ifdef GUROBI_USE
/**
 * Uses gurobi to solve for the worst case response subject to a weighted L1 constraint
 *
 * min_p  p^T z
 * s.t.   1^T p = 1^T pbar
 *        p >= 0
 *        ||p - pbar||_{1,w} <= xi
 *
 * The linear program formulation is as follows:
 *
 * min_{p,l} p^T z
 * s.t.   1^T p = 1^T pbar
 *        p >= 0
 *        p - pbar <= l
 *        pbar - p <= l
 *        w^T l <= xi
 *        l >= 0
 *
 *  @param wi Weights. Optional, all 1 if not provided.
 * @return Objective value and the optimal solution
 */
std::pair<numvec, double> worstcase_l1_w_gurobi(const GRBEnv& env, const numvec& z, const numvec& pbar, const numvec& wi, double xi){
    const size_t nstates = z.size();
    assert(nstates == pbar.size());
    assert(wi.empty() || nstates == wi.size());

    numvec ws;
    if(wi.empty()) ws = numvec(nstates, 1.0);
    const numvec& w = wi.empty() ? ws : wi;

    prec_t pbar_sum = accumulate(pbar.cbegin(), pbar.cend(), 0.0);

    GRBModel model = GRBModel(env);

    // Probabilities
    auto p = std::unique_ptr<GRBVar[]>(model.addVars(numvec(nstates,0.0).data(), nullptr,
                               nullptr,
                               std::vector<char>(nstates,GRB_CONTINUOUS).data(),
                               nullptr, nstates));
    // Element-wise errors
    auto l = std::unique_ptr<GRBVar[]>(model.addVars(numvec(nstates,0.0).data(), nullptr,
                               nullptr,
                               std::vector<char>(nstates,GRB_CONTINUOUS).data(),
                               nullptr, nstates));

    // constraint: 1^T p = 1
    GRBLinExpr ones;
    ones.addTerms(numvec(nstates,1.0).data(),p.get(), nstates);
    model.addConstr(ones, GRB_EQUAL, pbar_sum);

    // constraint: w^T l <= xi
    GRBLinExpr weights;
    weights.addTerms(w.data(),l.get(), nstates);
    model.addConstr(weights, GRB_LESS_EQUAL, xi);

    // constraints: p - pbar <= l (p - l <= pbar) and
    //              pbar - p <= l (l - p <= -pbar)
    for(size_t idstate = 0; idstate < nstates; idstate++){
        model.addConstr( p[idstate] - l[idstate] <=  pbar[idstate]);
        model.addConstr(-l[idstate] - p[idstate] <= -pbar[idstate]);
    }

    // objective p^T z
    GRBLinExpr objective;
    objective.addTerms(z.data(), p.get(), nstates);
    model.setObjective(objective, GRB_MINIMIZE);

    // solve
    model.optimize();

    // retrieve probability values
    numvec p_result(nstates);
    for(size_t i = 0; i < nstates; i++){
        p_result[i] = p[i].get(GRB_DoubleAttr_X);
    }

    // get optimal objective value
    double objective_value = model.get(GRB_DoubleAttr_ObjVal);

    return make_pair(move(p_result), objective_value);
}

/**
 * @brief Computes the worst case probability distribution subject to a wasserstein constraint
 *
 * min_{p, lambda} p^T z
 * s.t. 1^T p = 1
 *      p >= 0
 *      sum_j lambda_ij = pbar_i
 *      sum_i lambda_ij = p_j
 *      lambda_ij >= 0
 *
 * @param env Linear program envorinment to prevent repetitive initialization
 * @param z Objective value
 * @param pbar Reference probability distribution
 * @param dst Matrix of distances (or costs) when moving the distribution weights
 * @param xi Size of the ambiguity set
 * @return Worst-case distribution and the objective value
 */
std::pair<numvec, double> worstcase_wasserstein_gurobi(const GRBEnv& env, const numvec& z, const numvec& pbar, const numvecvec& dst, double xi){
    GRBModel model = GRBModel(env);

    size_t nstates = z.size();
    assert(nstates == z.size());
    assert(pbar.size() == nstates);
    assert(dst.size() == nstates);

    auto p = std::unique_ptr<GRBVar[]>(model.addVars(nullptr, nullptr,
                                    nullptr,
                                    std::vector<char>(nstates,GRB_CONTINUOUS).data(),
                                    nullptr, int(nstates)));

    std::vector<std::vector<GRBVar>> lambda(nstates, std::vector<GRBVar>(nstates));

    {
        GRBLinExpr wSum;
        for (size_t i = 0; i < nstates; ++i) {
            model.addConstr(p[i], GRB_GREATER_EQUAL, 0.0, "W_" + std::to_string(i) + "_nonNegative");
            p[i].set(GRB_StringAttr_VarName, "W_" + std::to_string(i));
            wSum += p[i];
        }
        model.addConstr(wSum, GRB_EQUAL, 1.0);
    }

    // create lambda variables
    for (size_t i = 0; i < nstates; ++i) {
        for (size_t j = 0; j < nstates; ++j) {
            lambda[i][j] = model.addVar(0.0, GRB_INFINITY, 0.0, GRB_CONTINUOUS,
                                           "lambda_w_" + std::to_string(i) + "_" + std::to_string(j));
        }
    }


    for (size_t i = 0; i < nstates; ++i) {
        GRBLinExpr psum;
        for (size_t j = 0; j < nstates; ++j) {
            psum += lambda[i][j];
        }
        model.addConstr(psum, GRB_EQUAL, p[i]);
    }

    for (size_t j = 0; j < nstates; ++j) {
        GRBLinExpr pbarsum;
        for (size_t i = 0; i < nstates; ++i) {
            pbarsum += lambda[i][j];
        }
        model.addConstr(pbarsum, GRB_EQUAL, pbar[j]);
    }

    {
        GRBLinExpr distance;
        for (size_t i = 0; i < nstates; ++i) {
            for (size_t j = 0; j < nstates; ++j) {
                distance += lambda[i][j] * dst[i][j];
            }
        }
        model.addConstr(distance, GRB_LESS_EQUAL, xi);
    }

    // objective value
    GRBLinExpr obj_w;
    obj_w.addTerms(z.data(), p.get(), int(nstates));
    model.setObjective(obj_w, GRB_MINIMIZE);

    model.optimize();
    //model_w.write("./debug_wass_w.lp");

    double objective_w = obj_w.getValue();
    numvec w_result(nstates);

    for (size_t i = 0; i < pbar.size(); i++) {
        w_result[i] = p[i].get(GRB_DoubleAttr_X);
        std::cout << "W[" << i << "]: " << w_result[i] << "\t\t";
    }

    return {move(w_result), objective_w};
}


/**
 * @brief worstcase_l1_w_gurobi Uses gurobi to solve for the worst case response subject to a weighted L1 constraint
 *
 * min_p  p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        ||p - pbar||_{2,w} <= xi
 *
 * The linear program formulation is as follows:
 *
 * min_{p,l} p^T z
 * s.t.   1^T p = 1
 *        p >= 0
 *        p - pbar <= l
 *        pbar - p <= l
 *        l^T diaq(w^2) l <= xi^2
 *        l >= 0
 *
 *  @param wi Weights. Optional, all 1 if not provided.
 * @return Objective value and the optimal solution
 */
std::pair<numvec, double> worstcase_l2_w_gurobi(const GRBEnv& env, const numvec& z, const numvec& pbar, const numvec& wi, double xi){
    const size_t nstates = z.size();
    assert(nstates == pbar.size());
    assert(wi.empty() || nstates == wi.size());

    numvec ws;
    if(wi.empty()) ws = numvec(nstates, 1.0);
    const numvec& w = wi.empty() ? ws : wi;


    GRBModel model = GRBModel(env);

    // Probabilities
    auto p = std::unique_ptr<GRBVar[]>(model.addVars(numvec(nstates,0.0).data(), nullptr,
                               nullptr,
                               std::vector<char>(nstates,GRB_CONTINUOUS).data(),
                               nullptr, nstates));
    // Element-wise errors
    auto l = std::unique_ptr<GRBVar[]>(model.addVars(numvec(nstates,0.0).data(), nullptr,
                               nullptr,
                               std::vector<char>(nstates,GRB_CONTINUOUS).data(),
                               nullptr, nstates));

    // constraint: 1^T p = 1
    GRBLinExpr ones;
    ones.addTerms(numvec(nstates,1.0).data(),p.get(), nstates);
    model.addConstr(ones, GRB_EQUAL, 1.0);

    // constraint: l^T W l <= xi^2
    numvec wsquared(w.size());
    transform(w.cbegin(), w.cend(), wsquared.begin(), [](double iw){return pow(iw,2);});
    GRBQuadExpr weights;
    weights.addTerms(wsquared.data(),l.get(), l.get(), nstates);
    model.addQConstr(weights, GRB_LESS_EQUAL, pow(xi,2));



    // constraints: p - pbar <= l (p - l <= pbar) and
    //              pbar - p <= l (l - p <= -pbar)
    for(size_t idstate = 0; idstate < nstates; idstate++){
        model.addConstr( p[idstate] - l[idstate] <=  pbar[idstate]);
        model.addConstr(-l[idstate] - p[idstate] <= -pbar[idstate]);
    }

    // objective p^T z
    GRBLinExpr objective;
    objective.addTerms(z.data(), p.get(), nstates);
    model.setObjective(objective, GRB_MINIMIZE);

    // solve
    model.optimize();

    // retrieve probability values
    numvec p_result(nstates);
    for(size_t i = 0; i < nstates; i++){
        p_result[i] = p[i].get(GRB_DoubleAttr_X);
    }

    // get optimal objective value
    double objective_value = model.get(GRB_DoubleAttr_ObjVal);

    return make_pair(move(p_result), objective_value);
}

#endif


}

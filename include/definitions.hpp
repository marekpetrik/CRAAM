#pragma once

#include "config.hpp"
#include <vector>

using namespace std;

namespace craam {

/** Default precision used throughout the code.*/
typedef double prec_t;

/** Default numericalk vector */
typedef vector<prec_t> numvec; // TODO: switch to valarray or an atlas array

/** Default index vector */
typedef vector<long> indvec;

/** Default solution precision */
const prec_t SOLPREC = 0.0001;
/** Default number of iterations */
const unsigned long MAXITER = 100000;

/** Function representing the constraints on nature. The inputs
    are the q-values z, the reference distribution q, and the threshold t.
    The function returns the worst-case solution and the objective value. */
typedef pair<vector<prec_t>,prec_t> (*NatureConstr)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t);

template <typename T> vector<size_t> sort_indexes(vector<T> const& v);
template <typename T> vector<size_t> sort_indexes_desc(vector<T> const& v);

pair<vector<prec_t>,prec_t> worstcase_l1(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t);

}

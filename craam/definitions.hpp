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

#include "config.hpp"

#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <string>
#include <assert.h>


#ifdef IS_DEBUG
// TODO: this is DEBUG ONLY
#include <iostream>
#include <string>
#endif

namespace craam {

using namespace std;

/** Default precision used throughout the code.*/
using prec_t = double;

/** Default numerical vector */
using numvec = vector<prec_t>;

/** Default index vector */
using indvec = vector<long>;

/** Probability list */
using prob_list_t = vector<prec_t>;

/** Probability matrix */
using prob_matrix_t = vector<prob_list_t>;

/** Pair of a vector and a scalar */
using vec_scal_t = pair<numvec, prec_t> ;

/** Tuple of a index, vector and a scalar */
using ind_vec_scal_t = tuple<prec_t, numvec, prec_t> ;

/** Default solution precision */
constexpr prec_t SOLPREC = 0.0001;

/** Default number of iterations */
constexpr unsigned long MAXITER = 100000;

/// Numerical threshold
constexpr prec_t THRESHOLD = 1e-5;


#ifdef IS_DEBUG
/** This is a useful functionality for debugging.  */
template<class T>
std::ostream & operator<<(std::ostream &os, const std::vector<T>& vec)
{
    for(const auto& p : vec){
        cout << p << " ";
    }
    return os;
}
#endif


/** \brief Sort indices by values in ascending order
 *
 * \param v List of values
 * \return Sorted indices
 */
template <typename T> 
inline
vector<size_t> sort_indexes(vector<T> const& v){
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});
    return idx;
}


/** \brief Sort indices by values in descending order
 *
 * \param v List of values
 * \return Sorted indices
 */
template <typename T> 
inline
vector<size_t> sort_indexes_desc(vector<T> const& v){
    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;
    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}


}

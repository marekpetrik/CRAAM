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
#include "definitions.hpp"

// The file includes methods for fast solutions of various optimization problems

namespace craam {

/**
Computes the solution of:
min_p   p^T * z
s.t.    ||p - q|| <= t
        1^T p = 1
        p >= 0

Notes
-----
This implementation works in O(n log n) time because of the sort. Using
quickselect to choose the right quantile would work in O(n) time.

This function does not check whether the probability distribution sums to 1.
**/
inline
pair<numvec,prec_t> worstcase_l1(numvec const& z, numvec const& q, prec_t t){
    assert(*min_element(q.cbegin(), q.cend()) >= - THRESHOLD);
    assert(*max_element(q.cbegin(), q.cend()) <= 1 + THRESHOLD);
    assert(z.size() > 0);
    assert(t >= 0.0 && t <= 2.0);
    assert(z.size() == q.size());

    size_t sz = z.size();

    vector<size_t> smallest = sort_indexes<prec_t>(z);
    numvec o(q);

    auto k = smallest[0];
    auto epsilon = min(t/2, 1-q[k]);

    o[k] += epsilon;

    auto i = sz - 1;
    while(epsilon > 0){
        k = smallest[i];
        auto diff = min( epsilon, o[k] );
        o[k] -= diff;
        epsilon -= diff;
        i -= 1;
    }

    auto r = inner_product(o.begin(),o.end(),z.begin(), (prec_t) 0.0);
    return make_pair(move(o),r);
}

}

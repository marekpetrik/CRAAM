#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <assert.h>

#include "definitions.hpp"

using namespace std;

namespace craam {

template <typename T> vector<size_t> sort_indexes(vector<T> const& v) {
    /** \brief Sort values by indices in ascending order
     *
     * \param v List of values
     * \return Sorted indices
     */

    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] < v[i2];});

    return idx;
}

template <typename T> vector<size_t> sort_indexes_desc(vector<T> const& v)
{
    /** \brief Sort values by indices in descending order
     *
     * \param v List of values
     * \return Sorted indices
     */

    // initialize original index locations
    vector<size_t> idx(v.size());
    for (size_t i = 0; i != idx.size(); ++i) idx[i] = i;

    // sort indexes based on comparing values in v
    sort(idx.begin(), idx.end(), [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

    return idx;
}

pair<vector<prec_t>,prec_t> worstcase_l1(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t){
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

    assert(*min_element(q.begin(), q.end()) >= 0 && *max_element(q.begin(), q.end()) <= 1);

    if(z.size() <= 0){
        throw invalid_argument("empty arguments");
    }
    if(t < 0.0 || t > 2.0){
        throw invalid_argument("incorrect threshold");
    }
    if(z.size() != q.size()){
        throw invalid_argument("parameter dimensions do not match");
    }


    size_t sz = z.size();

    vector<size_t> smallest = sort_indexes<prec_t>(z);
    vector<prec_t> o(q);

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

    return make_pair(o,r);
}


}

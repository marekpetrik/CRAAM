#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>

using namespace std;

typedef double prec_t;

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

template <typename T> vector<size_t> sort_indexes_desc(vector<T> const& v) {
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

#endif // DEFINITIONS_H

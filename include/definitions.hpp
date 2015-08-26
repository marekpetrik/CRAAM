#ifndef DEFINITIONS_H
#define DEFINITIONS_H

#include <vector>

using namespace std;

namespace craam {

typedef double prec_t;

template <typename T> vector<size_t> sort_indexes(vector<T> const& v);

template <typename T> vector<size_t> sort_indexes_desc(vector<T> const& v);

pair<vector<prec_t>,prec_t> worstcase_l1(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t);

}
#endif // DEFINITIONS_H

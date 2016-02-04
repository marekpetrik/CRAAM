#pragma once

#include "config.hpp"
#include <vector>

#ifdef IS_DEBUG
// TODO: this is DEBUG ONLY 
#include <iostream> 
#include <string>
#endif

using namespace std;

namespace craam {

/** Default precision used throughout the code.*/
typedef double prec_t;

/** Default numericalk vector */
typedef vector<prec_t> numvec; // TODO: switch to valarray or an atlas array

/** Default index vector */
typedef vector<long> indvec;


/** Function representing the constraints on nature. The inputs
    are the q-values z, the reference distribution q, and the threshold t.
    The function returns the worst-case solution and the objective value. */
typedef pair<vector<prec_t>,prec_t> (*NatureConstr)(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t);

template <typename T> vector<size_t> sort_indexes(vector<T> const& v);
template <typename T> vector<size_t> sort_indexes_desc(vector<T> const& v);

pair<vector<prec_t>,prec_t> worstcase_l1(vector<prec_t> const& z, vector<prec_t> const& q, prec_t t);

}


#ifdef IS_DEBUG
/**
 * This is a useful functionality for debugging
 */
template<class T>
std::ostream & operator<<(std::ostream &os, const std::vector<T>& vec)
{
    for(const auto& p : vec){
        cout << p << " ";
    }
    return os;
}
#endif
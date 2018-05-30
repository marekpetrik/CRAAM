#include <Rcpp.h>
#include <tuple>
#include "craam/RMDP.hpp"
#include "craam/optimization/optimization.hpp"

/// Computes the maximum distribution subject to L1 constraints
// [[Rcpp::export]]
Rcpp::List worstcase_l1(Rcpp::NumericVector z, Rcpp::NumericVector q, double t){
    // resulting probability
    craam::numvec p;
    // resulting objective value
    double objective;

    craam::numvec vz(z.begin(), z.end()), vq(q.begin(), q.end());
    std::tie(p,objective) = craam::worstcase_l1(vz,vq,t);

    Rcpp::List result;
    result["p"] = Rcpp::NumericVector(p.cbegin(), p.cend());
    result["obj"] = objective;

    return result;
}

MDP mdp_from_dataframe(const Rcpp::DataFrame data){
    // idstatefrom, idaction, idstateto, probability, reward

}


// [[Rcpp::export]]
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, Rcpp::String algorithm){
    Rcpp::List result;

    result["x"] = algorithm;
    return result;
}

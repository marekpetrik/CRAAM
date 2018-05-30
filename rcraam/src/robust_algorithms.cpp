#include <Rcpp.h>
#include <tuple>
#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"
#include "craam/optimization/optimization.hpp"

#include <stdexcept>

using namespace craam;

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

craam::MDP mdp_from_dataframe(const Rcpp::DataFrame& data){
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"],
                        idaction = data["idaction"],
                        idstateto = data["idstateto"];
    Rcpp::NumericVector probability = data["probability"],
                        reward = data["reward"];

    size_t n = data.nrow();
    craam::MDP m;

    for(size_t i = 0; i < n; i++){
        craam::add_transition(m, idstatefrom[i], idaction[i], idstateto[i], probability[i], reward[i]);
    }

    return m;

}


// [[Rcpp::export]]
Rcpp::List solve_mdp(Rcpp::DataFrame mdp, double discount, Rcpp::String algorithm){
    MDP m = mdp_from_dataframe(mdp);

    int iterations = 1000;
    double precision = 0.01;

    algorithms::DeterministicSolution sol;
    if(algorithm == "mpi") {
        sol = algorithms::solve_mpi(m,discount,numvec(0),indvec(0),
                                    iterations,precision,iterations,precision);
    } else if(algorithm == "vi") {
        sol = algorithms::solve_vi(m,discount,numvec(0),indvec(0),
                                   iterations,precision);
    } else {
        throw std::invalid_argument("Unknown solver type.");
    }

    Rcpp::List result;
    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["policy"] = move(sol.policy);
    result["valuefunction"] = move(sol.valuefunction);
    return result;
}

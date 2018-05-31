#include <Rcpp.h>
#include <tuple>
#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"
#include "craam/algorithms/robust_values.hpp"
#include "craam/optimization/optimization.hpp"
#include "craam/algorithms/nature_response.hpp"
#include "craam/algorithms/nature_declarations.hpp"

#include <stdexcept>

using namespace craam;

/** Computes the maximum distribution subject to L1 constraints */
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

/**
 * Parses a data frame  to an mdp
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idstateto, reward, probability.
 *              Multiple state-action-state rows have summed probabilities and averaged rewards.
 *
 * @returns Corresponding MDP definition
 */
craam::MDP mdp_from_dataframe(const Rcpp::DataFrame& data){
    // idstatefrom, idaction, idstateto, probability, reward
    Rcpp::IntegerVector idstatefrom = data["idstatefrom"],
                        idaction = data["idaction"],
                        idstateto = data["idstateto"];
    Rcpp::NumericVector probability = data["probability"],
                        reward = data["reward"];

    size_t n = data.nrow();
    MDP m;

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
        Rcpp::stop("Unknown solver type.");
    }

    Rcpp::List result;
    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;
    result["policy"] = move(sol.policy);
    result["valuefunction"] = move(sol.valuefunction);
    return result;
}

/**
 * Parses a data frame definition of values that correspond to states and
 * actions.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 3 comlumns, idstate, idaction, value. Here, idstate and idaction
 *              determine which value should be set.
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not specify anything for the state action pair
 *
 * @returns A vector over states with an inner vector of actions
 */
vector<numvec> parse_sa_values(const MDP& mdp, const Rcpp::DataFrame& frame, double def_value = 0){

    vector<numvec> result(mdp.size());
    for(long i = 0; i < mdp.size(); i++){
        result[i] = numvec(mdp[i].size(), def_value);
    }

    Rcpp::IntegerVector idstates = frame["idstate"],
                        idactions = frame["idaction"];
    Rcpp::NumericVector values = frame["value"];

    for(long i = 0; i < idstates.size(); i++){
        long idstate = idstates[i],
             idaction = idactions[i];

        if(idstate < 0) Rcpp::stop("idstate must be non-negative");
        if(idstate > mdp.size()) Rcpp::stop("idstate must be smaller than the number of MDP states");
        if(idaction < 0) Rcpp::stop("idaction must be non-negative");
        if(idaction > mdp[idstate].size()) Rcpp::stop("idaction must be smaller than the number of actions for the corresponding state");

        double value = values[i];
        result[idstate][idaction] = value;
    }

    return result;
}

/**
 * Parses a data frame definition of values that correspond to starting states, actions,
 * ans taget states.
 *
 * Also checks whether the values passed are consistent with the MDP definition.
 *
 * @param mdp The definition of the MDP to know how many states and actions there are.
 * @param frame Dataframe with 3 comlumns, idstatefrom, idaction, idstateto, value.
 *              Here, idstate(from,to) and idaction determine which value should be set
 *              Only the last value is used if multiple rows are present.
 * @param def_value The default value for when frame does not specify anything for the state action pair
 *
 * @returns A vector over states, action, with an inner vector of actions
 */
vector<vector<numvec>> parse_sas_values(const MDP& mdp, const Rcpp::DataFrame& frame, double def_value = 0){

    vector<vector<numvec>> result(mdp.size());
    for(long i = 0; i < mdp.size(); i++){
        for(long j = 0; j < mdp[i].size(); j++){
            result[i][j] = numvec(mdp[i][j].size(), def_value);
        }
    }

    Rcpp::IntegerVector idstatesfrom = frame["idstatefrom"],
                        idactions = frame["idaction"],
                        idstatesto = frame["idstateto"];
    Rcpp::NumericVector values = frame["value"];

    for(long i = 0; i < idstatesfrom.size(); i++){
        long idstatefrom = idstatesfrom[i],
             idstateto = idstatesto[i],
             idaction = idactions[i];

        if(idstatefrom < 0) Rcpp::stop("idstatefrom must be non-negative");
        if(idstatefrom > mdp.size()) Rcpp::stop("idstatefrom must be smaller than the number of MDP states");
        if(idaction < 0) Rcpp::stop("idaction must be non-negative");
        if(idaction > mdp[idstatefrom].size()) Rcpp::stop("idaction must be smaller than the number of actions for the corresponding state");
        if(idstateto < 0) Rcpp::stop("idstateto must be non-negative");
        if(idstateto > mdp[idstatefrom][idaction].size()) Rcpp::stop("idstateto must be smaller than the number of positive transition probabilites");

        double value = values[i];
        result[idstatefrom][idaction][idstateto] = value;
    }

    return result;
}

/**
 * Parses the name and the parameter of the provided nature
 */
algorithms::SANature parse_nature_sa(const MDP& mdp, const string& nature, SEXP nature_par){
    if(nature == "l1u"){
        return algorithms::nats::robust_l1u(Rcpp::as<double>(nature_par));
    }
    if(nature == "l1"){
        vector<numvec> values = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(nature_par), 0.0);
        return algorithms::nats::robust_l1(values);
    }
    if(nature == "l1w"){
        Rcpp::List par = Rcpp::as<Rcpp::List>(nature_par);
        auto budgets = parse_sa_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["budgets"]),0.0);
        auto weights = parse_sas_values(mdp, Rcpp::as<Rcpp::DataFrame>(par["weights"]), 1.0);
        return algorithms::nats::robust_l1w(budgets, weights);
    }else{
        Rcpp::stop("unknown nature");
    }
}

// [[Rcpp::export]]
Rcpp::List rsolve_mdp_sa(Rcpp::DataFrame mdp, double discount,
                         Rcpp::String nature, SEXP nature_par,
                         Rcpp::String algorithm){

    MDP m = mdp_from_dataframe(mdp);

    int iterations = 1000;
    double precision = 0.01;

    algorithms::SARobustSolution sol;
    algorithms::SANature natparsed = parse_nature_sa(m, nature, nature_par);
    if(algorithm == "mpi") {
        sol = algorithms::rsolve_mpi(m,discount,natparsed,numvec(0),indvec(0),
                                    iterations,precision,iterations,precision);
    } else if(algorithm == "vi") {
        sol = algorithms::rsolve_vi(m,discount,natparsed,numvec(0),indvec(0),
                                   iterations,precision);
    } else {
        Rcpp::stop("Unknown solver type.");
    }

    Rcpp::List result;
    result["iters"] = sol.iterations;
    result["residual"] = sol.residual;
    result["time"] = sol.time;

    auto [dec_pol, nat_pol] = unzip(sol.policy);
    result["policy"] = move(dec_pol);
    result["policy.nature"] = move(nat_pol);
    result["valuefunction"] = move(sol.valuefunction);
    return result;
}


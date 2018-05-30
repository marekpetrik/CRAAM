/*#include <Rcpp.h>
#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"

#include "craam/fastopt.hpp"
#include <iostream>

using MDP = craam::MDP;

void from_dataframe(MDP* mdp, const Rcpp::DataFrame& data){

    int posStateFrom = data.findName("idstatefrom");
    int posStateTo = data.findName("idstateto");
    int posAction = data.findName("idaction");
    int posReward = data.findName("reward");
    int posProbability = data.findName("probability");

    const Rcpp::IntegerVector& statefrom = data[posStateFrom];
    const Rcpp::IntegerVector& action = data[posAction];
    const Rcpp::IntegerVector& stateto = data[posStateTo];
    const Rcpp::NumericVector& probability = data[posProbability];
    const Rcpp::NumericVector& reward = data[posReward];

    for(int row = 0; row  < data.nrows(); row++){
        add_transition(*mdp, statefrom[row],
                             action[row],
                             stateto[row],
                             probability[row],
                             reward[row]);

    }

}


Rcpp::List solve_mpi(MDP* mdp, double discount){
    auto sol = craam::algorithms::solve_mpi(*mdp, discount);
    Rcpp::List res;
    res["valuefunction"] = sol.valuefunction;
    res["policy"] = sol.policy;
    return res;
}

RCPP_MODULE(class_MDP) {
    using namespace Rcpp;

    // expose the class as MDP in R
    class_<MDP>("MDP")

    // expose the default constructor
    .constructor()

    // expose some internal methods
    .method("size", &MDP::size)

    // expose auxilliary methods
    .method("from_dataframe", &from_dataframe)
    .method("solve_mpi", &solve_mpi);

}


*/

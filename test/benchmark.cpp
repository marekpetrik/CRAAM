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

#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"
#include "craam/Samples.hpp"
#include "craam/algorithms/nature_response.hpp"
#include "craam/algorithms/robust_values.hpp"

// library for processing command-line options
#include "cxxopts/cxxopts.hpp"

// lirary for reading csv files
#include "fccp/csv.h"

#include <chrono>
#include <iostream>
#include <fstream>


using namespace std;
using namespace craam;

enum class Solver{
VI, MPI
};

void solve_mdp(const cxxopts::Options& options, Solver solver){
    cout << "Loading ... " << endl;

    // open file
    ifstream ifs; ifs.open(options["input"].as<string>());
    if(!ifs.is_open()){
        cout << "Failed to open the input file." << endl;
        terminate();
    }

    // solve the MDP
    MDP mdp;
    from_csv(mdp,ifs,true,false);
    ifs.close();

    // parse computation-related options
    auto iterations = options["iterations"].as<unsigned long>();
    prec_t discount = options["discount"].as<prec_t>();
    prec_t precision = options["precision"].as<prec_t>();

    const auto ambiguity = options["ambiguity"].as<std::string>();

    cout << "Computing solution ... " << endl;
    auto start = std::chrono::high_resolution_clock::now();

    // output values
    int iters; prec_t residual; indvec policy; numvec valuefunction;

    if(ambiguity.empty()) {
        algorithms::DeterministicSolution sol;
        if(solver == Solver::MPI) {
            sol = algorithms::solve_mpi(mdp,discount,numvec(0),indvec(0),
                                        iterations,precision,iterations,0.5);
        } else if(solver == Solver::VI) {
            sol = algorithms::solve_vi(mdp,discount,numvec(0),indvec(0),
                                       iterations,precision);
        } else {
            throw invalid_argument("Unknown solver type.");
        }
        iters = sol.iterations;
        residual = sol.residual;
        policy = move(sol.policy);
        valuefunction = move(sol.valuefunction);

    }
    else if(ambiguity == "L1") {
        algorithms::SARobustSolution sol;

        prec_t budget = options["budget"].as<prec_t>();
        numvecvec budgets = map_sa<prec_t>(mdp,
            [budget](const RegularState&,const RegularAction&){return budget;});

        if(solver == Solver::MPI) {
            sol = algorithms::rsolve_mpi(mdp, discount, algorithms::nats::robust_l1(budgets), numvec(0), indvec(0),
                                        iterations, precision, 0.5, precision);
        } else if(solver == Solver::VI) {
            sol = algorithms::rsolve_vi(mdp, discount, algorithms::nats::robust_l1(budgets), numvec(0), indvec(0),
                                       iterations, precision);
        } else {
            throw invalid_argument("Unknown solver type.");
        }
        iters = sol.iterations;
        residual = sol.residual;
        policy = unzip(sol.policy).first;
        valuefunction = move(sol.valuefunction);
    }
    #ifdef GUROBI_USE
    else if(ambiguity == "L1g") {
        algorithms::SARobustSolution sol;

        prec_t budget = options["budget"].as<prec_t>();
        numvecvec budgets = map_sa<prec_t>(mdp,
            [budget](const RegularState&,const RegularAction&){return  budget;});

        if(solver == Solver::MPI) {
            sol = algorithms::rsolve_mpi(mdp, discount, algorithms::nats::robust_l1w_gurobi(budgets),
                                        numvec(0), indvec(0),
                                        iterations, precision, 0.5, precision);
        } else if(solver == Solver::VI) {
            sol = algorithms::rsolve_vi(mdp, discount, algorithms::nats::robust_l1w_gurobi(budgets),
                                        numvec(0), indvec(0),
                                        iterations, precision);
        } else {
            throw invalid_argument("Unknown solver type.");
        }
        iters = sol.iterations;
        residual = sol.residual;
        policy = unzip(sol.policy).first;
        valuefunction = move(sol.valuefunction);
    }
    #endif
    else {
        throw invalid_argument("Unknown uncertainty set type.");
    }

    auto finish = std::chrono::high_resolution_clock::now();
    cout << "Done computing." << endl;
    cout << "Iterations: " << iters <<  ", Residual: " << residual << endl;
    std::cout << "Duration:  *** " << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() << "ms *** \n";

    // only do this when an output file has been provided
    if(options.count("output") > 0){
        cout << "Writing output ... " << endl;
        ofstream ofs;
        ofs.open(options["output"].as<string>());
        if(!ofs.is_open()){
            cout << "Could not open the output file for writing" << endl;
            terminate();
        }
        ofs << "idstate,idaction,value" << endl;
        for(size_t i = 0; i < mdp.state_count(); i++){
            ofs << i << "," << policy[i] << "," << valuefunction[i] << endl;
        }
        ofs.close();
    }
    cout << "Done." << endl;
}

void build_mdp(const cxxopts::Options& options){

    // load samples (reward is received after state and action in the same line)
    io::CSVReader<4> in(options["input"].as<string>());
    in.read_header(io::ignore_extra_column, "step", "idstate", "idaction", "reward");

    cout << "Loading samples ... " << endl;
    craam::msen::DiscreteSamples samples;

    // values from the last sample
    int last_step, last_state, last_action;
    double last_reward;

    if(!in.read_row(last_step, last_state, last_action, last_reward)){
        cout << "Only one sample ... cannot do anything with that. " << endl;
        terminate();
    }

    int run = 0;
    // values from the current sample
    int step, state, action;
    double reward;

    while (in.read_row(step, state, action, reward)){
        if(step == last_step + 1){
            // this means we are not at the boundary between episodes, we can add the sample
            samples.add_sample(last_state, last_action, state, last_reward, 1.0, step, run);
        }else{ // else we just skip the sample
            run++;
        }
        last_step = step; last_state = state; last_action = action; last_reward = reward;
    }

    cout << "Loaded " <<  samples.size() << " samples " << endl;

    cout << "Building MDP ... " << endl;
    craam::msen::SampledMDP smdp;
    smdp.add_samples(samples);

    if(options.count("output") > 0){
         cout << "Writing output ... " << endl;
         to_csv_file(*smdp.get_mdp(), options["output"].as<string>());
    }
}

int main(int argc, char * argv []){

    cxxopts::Options options("craam", "Fast command-line solver for (robust) MDPs");

    options.add_options()
        ("h,help", "Display help message.")
        ("i,input", "Input file, MDP description or samples (csv file)", cxxopts::value<string>())
        ("o,output", "Policy and value output (csv file)", cxxopts::value<string>() )
        ("m,method", "Solution method: MPI - Modified Policy Iteration, VI - Value Iteration, MDP - Construct from samples",
                                                                            cxxopts::value<string>()->default_value("MPI"))
        ("d,discount", "Discount factor", cxxopts::value<double>()->default_value("0.9"))
        ("e,precision", "Maximum residual", cxxopts::value<double>()->default_value("0.0001"))
        ("l,iterations", "Maximum number of iterations", cxxopts::value<unsigned long>()->default_value("2000"))
        ("b,budget", "Robustness budget", cxxopts::value<double>()->default_value("0.0"))
        ("u,ambiguity", "Type of ambiguity", cxxopts::value<string>());

    try {
        options.parse(argc,argv);
    } catch (const cxxopts::OptionException& oe) {
        cout << oe.what() << endl << endl << " *** usage *** " << endl;
        cout << options.help() << endl;
        terminate();
    }

    if(options["h"].as<bool>()){
        cout << options.help() << endl;
        terminate();
    }

    if(options.count("input") == 0){
        cout << "No input file provided. See usage (-h)." << endl;
        terminate();
    }

    const auto method = options["method"].as<std::string>();
    if(method == "MPI"){
        solve_mdp(options, Solver::MPI);
    }
    else if(method == "VI"){
        solve_mdp(options, Solver::VI);
    }
    else if(method == "MDP"){
        build_mdp(options);
    }
    else{
        cout << "Unknown method type: " << method << "." << endl;
        terminate();
    }

}

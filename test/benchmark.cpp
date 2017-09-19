#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"

// library for processing command-line options
#include "cxxopts/cxxopts.hpp"

#include <chrono>
#include <iostream>
#include <fstream>


using namespace std;
using namespace craam;

int main(int argc, char * argv []){

    cxxopts::Options options("craam", "Fast command-line solver for (robust) MDPs");

    options.add_options()
        ("h,help", "Display help message.")
        ("i,input", "MDP description (csv file)", cxxopts::value<string>())
        ("o,output", "Policy and value output (csv file)", cxxopts::value<string>() )
        ("d,discount", "Discount factor", cxxopts::value<double>()->default_value("0.9"))
        ("e,precision", "Maximum residual", cxxopts::value<double>()->default_value("0.0001"))
        ("l,iterations", "Maximum number of iterations", cxxopts::value<unsigned long>()->default_value("2000"));

    try {
        options.parse(argc,argv);
    } catch (cxxopts::OptionException oe) {
        cout << oe.what() << endl << endl << " *** usage *** " << endl;
        cout << options.help() << endl;
        return -1;
    }

    if(options["h"].as<bool>()){
        cout << options.help() << endl;
        return 0;
    }

    cout << "Loading ... " << endl;
    if(options.count("input") == 0){
        cout << "No input file provided. See usage (-h)." << endl;
        return -1;
    }

    // open file
    ifstream ifs; ifs.open(options["input"].as<string>());
    if(!ifs.is_open()){
        cout << "Failed to open the input file." << endl;
        return -1;
    }

    // solve the MDP
    MDP mdp;
    from_csv(mdp,ifs,true,false);
    ifs.close();

    // parse computation-related options
    auto iterations = options["iterations"].as<unsigned long>();
    prec_t discount = options["discount"].as<prec_t>();
    prec_t precision = options["precision"].as<prec_t>();

    cout << "Computing solution ... " << endl;
    auto start = std::chrono::high_resolution_clock::now();
    auto sol = algorithms::solve_mpi(mdp,discount,numvec(0),indvec(0),
                                     iterations,precision,iterations,precision);
    auto finish = std::chrono::high_resolution_clock::now();
    cout << "Done computing." << endl;
    cout << "Iterations: " << sol.iterations <<  ", Residual: " << sol.residual << endl;
    std::cout << "Duration:  *** " << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() << "ms *** \n";

    // only do this when an output file has been provided
    if(options.count("output") > 0){
        cout << "Writing output ... " << endl;
        ofstream ofs;
        ofs.open(options["output"].as<string>());
        if(!ofs.is_open()){
            cout << "Could not open the output file for writing" << endl;
            return -1;
        }
        ofs << "idstate,idaction,value" << endl;
        for(size_t i = 0; i < mdp.state_count(); i++){
            ofs << i << "," << sol.policy[i] << "," << sol.valuefunction[i] << endl;
        }
        ofs.close();
    }
    cout << "Done." << endl;
}

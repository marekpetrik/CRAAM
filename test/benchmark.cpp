#include "craam/RMDP.hpp"
#include "craam/modeltools.hpp"
#include "craam/algorithms/values.hpp"

#include <chrono>
#include <iostream>
#include <fstream>


using namespace std;
using namespace craam;

int main(int argc, char * argv []){

    if(argc != 2){
        cout << "Invalid execution parameters. Execute as: " << endl;
        cout << argv[0] << " mdp_file " << endl;
        return -1;
    }

    string filename = argv[1];

    cout << "loading" << endl;

    ifstream ifs;
    ifs.open(filename);
    if(!ifs.is_open()){
        cout << "file could not be open";
        return -1;
    }
    RMDP rmdp;
    from_csv(rmdp,ifs,true);
    ifs.close();

    cout << "running test" << endl;

    auto start = std::chrono::high_resolution_clock::now();

    //#include <gperftools/profiler.h>
    // library option -lprofiler
    //ProfilerStart("mpi.prof");

    auto&& sol = algorithms::solve_mpi(rmdp,0.999,numvec(0),indvec(0),2000,0.0001,2000,0.0001);

    auto finish = std::chrono::high_resolution_clock::now();

    cout << "done." << endl;
    cout << "Iterations: " << sol.iterations <<  ", Residual: " << sol.residual << endl;

    std::cout << "Duration:  *** " << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() << "ms *** \n";
}


#include <chrono>
#include <iostream>
#include <fstream>

#include "RMDP.hpp"

using namespace std;
using namespace craam;

int main(int argc, char * argv []){

    if(argc != 2){
        cout << "Invalid execution parameters. Execute as: " << endl;
        cout << "raam benchmark_file " << endl;
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
    auto rmdp = RMDP::transitions_from_csv(ifs,true);
    ifs.close();

    cout << "running test" << endl;

    auto start = std::chrono::high_resolution_clock::now();

    //#include <gperftools/profiler.h>
    // library option -lprofiler
    //ProfilerStart("mpi.prof");

    vector<prec_t> value(rmdp->state_count());
    auto&& sol = rmdp->mpi_jac_rob(value,0.999,2000,0.0001,2000,0.0001);

    //ProfilerStop();

    auto finish = std::chrono::high_resolution_clock::now();

    cout << "done." <<endl;
    cout << "Iterations: " << sol.iterations <<  ", Residual: " << sol.residual << endl;

    std::cout << "Duration:  *** " << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() << "ms *** \n";
}

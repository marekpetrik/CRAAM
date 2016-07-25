#include "RMDP.hpp"
#include "modeltools.hpp"

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
    RMDP_D rmdp;
    from_csv(rmdp,ifs,true);
    ifs.close();

    cout << "running test" << endl;

    auto start = std::chrono::high_resolution_clock::now();

    //#include <gperftools/profiler.h>
    // library option -lprofiler
    //ProfilerStart("mpi.prof");

    vector<prec_t> value(rmdp.state_count(),0.0);
    auto&& sol = rmdp.mpi_jac(Uncertainty::Robust,0.999,value,2000,0.0001,2000,0.0001);

    //ProfilerStop();

    auto finish = std::chrono::high_resolution_clock::now();

    cout << "done." << endl;
    cout << "Iterations: " << sol.iterations <<  ", Residual: " << sol.residual << endl;

    std::cout << "Duration:  *** " << std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count() << "ms *** \n";
}


#include "Simulation.hpp"


using namespace std;


template <class DecState,class Action,class ExpState>
void Samples<DecState,Action,ExpState>::add_dec(const DecSample<DecState,Action,ExpState>& decsample){
    /**
     * \brief Adds a sample starting in a decision state
     */

     this->decsamples.push_back(decsample);
};

template <class DecState,class Action,class ExpState>
void Samples<DecState,Action,ExpState>::add_exp(const ExpSample<DecState,ExpState>& expsample){
    /**
     * \brief Adds a sample starting in an expectation state
     */

    this->expsamples.push_back(expsample);
};

#include <pair>

using namespace std;


struct ExpSample<DecState,ExpState> {
    /**
     * \brief Represents the transition from an expectation state to a
     * a decision state.
     */
    ExpState expstate_from;
    DecState decstate_to;
    prec_t weight;
    long step;
    long run;

    ExpSample(const ExpState& expstate_from, const DecState& decstate_to,
              prec_t weight, long step, long run){
        this->expstate_from = expstate_from;
        this->decstate_to = decstate_to;
        this->weight = weight;
        this->step = step;
        this->run = run;
    };
};

struct DecSample<DecState,Action,ExpState=pair<DecState,Action>> {
    /**
     * \brief Represents the transition from a decision state to an
     * expectation state.
     */
    DecState decstate_from;
    Action action;
    ExpState expstate_to;
    long step;
    long run;

    DecSample(const DecState& decstate_from, const Action& action,
              const ExpState& expstate_to, long step, long run){

        this->decstate_from = decstate_from;
        this->action = action;
        this->expstate_to = expstate_to;
        this->step = step;
        this->run = run;
    };
};

class Samples<DecState,Action,ExpState> {
    /**
     * \brief General representation of samples
     */


};

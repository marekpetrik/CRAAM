#include<utility>
#include<vector>
#include<memory>

#include "definitions.hpp"

using namespace std;


template <class DecState,class ExpState>
struct ExpSample {
    /**
     * \brief Represents the transition from an expectation state to a
     * a decision state.
     */
    const ExpState expstate_from;
    const DecState decstate_to;
    const prec_t reward;
    const prec_t weight;
    const long step;
    const long run;

    ExpSample(const ExpState& expstate_from, const DecState& decstate_to,
              prec_t reward, prec_t weight, long step, long run):
        expstate_from(expstate_from), decstate_to(decstate_to),
        reward(reward), weight(weight), step(step), run(run)
                  {};
};

template <class DecState,class Action,class ExpState=pair<DecState,Action>>
struct DecSample {
    /**
     * \brief Represents the transition from a decision state to an
     * expectation state.
     */
    const DecState decstate_from;
    const Action action;
    const ExpState expstate_to;
    const long step;
    const long run;

    DecSample(const DecState& decstate_from, const Action& action,
              const ExpState& expstate_to, long step, long run):
        decstate_from(decstate_from), action(action),
        expstate_to(expstate_to), step(step), run(run)
        {};
};

template <class DecState,class Action,class ExpState=pair<DecState,Action>>
class Samples {
    /**
     * \brief General representation of samples
     */

public:
    vector<DecSample<DecState,Action,ExpState>> decsamples;
    vector<DecState> initial;
    vector<ExpSample<DecState,ExpState>> expsamples;

public:

    void add_dec(const DecSample<DecState,Action,ExpState>& decsample){
        /**
        * \brief Adds a sample starting in a decision state
        */

        this->decsamples.push_back(decsample);
    };

    void add_initial(const DecState& decstate){
        /**
         * \brief Adds an initial state
         *
         */
         this->initial.push_back(decstate);

    };

    void add_exp(const ExpSample<DecState,ExpState>& expsample){
        /**
         * \brief Adds a sample starting in an expectation state
         */

        this->expsamples.push_back(expsample);
    }

};

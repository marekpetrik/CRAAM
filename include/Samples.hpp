#pragma once

#include <set>

#include "definitions.hpp"

using namespace std;

namespace craam{
namespace msen {

/**
Represents the transition from an expectation state to a
a decision state.

\tparam Sim Simulator class used to generate the sample. Only the members
            defining the types of DState, Action, and EState are necessary.
*/
template <class Sim>
struct ESample {

    const typename Sim::EState expstate_from;
    const typename Sim::DState decstate_to;
    const prec_t reward;
    const prec_t weight;
    const long step;
    const long run;

    ESample(const typename Sim::EState& expstate_from, const typename Sim::DState& decstate_to,
              prec_t reward, prec_t weight, long step, long run):
        expstate_from(expstate_from), decstate_to(decstate_to),
        reward(reward), weight(weight), step(step), run(run)
                  {};
};

/**
Represents the transition from a decision state to an expectation state.

\tparam Sim Simulator class used to generate the sample. Only the members
            defining the types of DState, Action, and EState are necessary.
 */
template <class Sim>
struct DSample {

    const typename Sim::DState decstate_from;
    const typename Sim::Action action;
    const typename Sim::EState expstate_to;
    const long step;
    const long run;

    DSample(const typename Sim::DState& decstate_from, const typename Sim::Action& action,
              const typename Sim::EState& expstate_to, long step, long run):
        decstate_from(decstate_from), action(action),
        expstate_to(expstate_to), step(step), run(run)
        {};
};

/**
General representation of samples.

\tparam Sim Simulator class used to generate the samples. Only the members
            defining the types of DState, Action, and EState are necessary.
 */
template <typename Sim>
class Samples {
public:
    vector<DSample<Sim>> decsamples;
    vector<typename Sim::DState> initial;
    vector<ESample<Sim>> expsamples;

public:
    void add_dec(const DSample<Sim>& decsample){
        /**
        Adds a sample starting in a decision state
        */
        this->decsamples.push_back(decsample);
    };

    void add_initial(const typename Sim::DState& decstate){
        /**
        Adds an initial state
        */
        this->initial.push_back(decstate);
    };

    void add_exp(const ESample<Sim>& expsample){
        /**
           Adds a sample starting in an expectation state
         */
        this->expsamples.push_back(expsample);
    };

    prec_t mean_return(prec_t discount){
        /**
        Computes the discounted mean return over all the
        samples

        \param discount Discount factor
        */

        prec_t result = 0;

        set<int> runs;

        for(const auto& es : expsamples){
           result += es.reward * pow(discount,es.step);
           runs.insert(es.run);
        }

        result /= runs.size();

        return result;
    };
};


/**
Constructs MDP from integer samples. In integer samples, each
decision state, expectation state, and action are identified
by an integer,
*/
class SampledMDP{


};

class SampledMDP_Exp{


};

}
}

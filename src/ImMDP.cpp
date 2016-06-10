#include "definitions.hpp"
#include "RMDP.hpp"
#include "ImMDP.hpp"

#include <algorithm>
#include <memory>

#include <iostream>
#include <iterator>

#include "cpp11-range-master/range.hpp"

using namespace std;
using namespace util::lang;

namespace craam{namespace impl{

template<typename T>
T max_value(vector<T> x){
    return (x.size() > 0) ? *max_element(x.begin(), x.end()) : -1;
}


void MDPI::check_parameters(const MDP& mdp, const indvec& state2observ,
                            const Transition& initial){

    // *** check consistency of provided parameters ***
    // check that the number of state2observ coefficients it correct
    if(mdp.state_count() !=  state2observ.size())
        throw invalid_argument("Number of observation indexes must match the number of states.");
    // check that the observation indexes are not negative
    if(state2observ.size() == 0)
        throw invalid_argument("Cannot have empty observations.");
    if(*min_element(state2observ.begin(), state2observ.end()) < 0)
        throw invalid_argument("Observation indexes must be non-negative");
    // check then initial transition
    if(initial.max_index() >= (long) mdp.state_count())
        throw invalid_argument("An initial transition to a non-existent state.");
    if(!initial.is_normalized())
        throw invalid_argument("The initial transition must be normalized.");

}

MDPI::MDPI(const shared_ptr<const MDP>& mdp, const indvec& state2observ,
           const Transition& initial)
            : mdp(mdp), state2observ(state2observ), initial(initial),
              obscount(1+max_value(state2observ)),
              action_counts(obscount, -1){

    check_parameters(*mdp, state2observ, initial);

    for(auto state : range((size_t) 0, mdp->state_count())){
        auto obs = state2observ[state];

        // check the number of actions
        auto ac = mdp->get_state(state).action_count();
        if(action_counts[obs] >= 0){
            if(action_counts[obs] != (long) ac){
                throw invalid_argument("Inconsistent number of actions: " + to_string(ac) +
                                       " instead of " + to_string(action_counts[obs]) +
                                       " in state " + to_string(state));
            }
        }else{
            action_counts[obs] = ac;
        }
    }
}

MDPI::MDPI(const MDP& mdp, const indvec& state2observ, const Transition& initial)
            : MDPI(make_shared<const MDP>(mdp),state2observ, initial){}


void MDPI::obspol2statepol(const indvec& obspol, indvec& statepol) const{
    assert(obspol.size() == (size_t) obscount);
    assert(mdp->state_count() == statepol.size());

    for(auto s : range((size_t)0, state_count())){
        statepol[s] = obspol[state2observ[s]];
    }
}

indvec MDPI::obspol2statepol(const indvec& obspol) const{
    indvec statepol(state_count());
    obspol2statepol(obspol, statepol);
    return statepol;
}


indvec MDPI::random_policy(random_device::result_type seed){

    indvec policy(obscount, -1);

    default_random_engine gen(seed);

    for(auto obs : range(0l, obscount)){
        auto ac = action_counts[obs];
        if(ac == 0)
            continue;

        uniform_int_distribution<int> dist(0,ac-1);
        policy[obs] = dist(gen);
    }

    return policy;
}

prec_t MDPI::total_return(const indvec& obspol, prec_t discount, prec_t precision) const {

    indvec&& statepol = obspol2statepol(obspol);
    indvec natpolicy(mdp->state_count(), (size_t) 0);

    auto&& sol = mdp->vi_jac_fix(discount, statepol, natpolicy, numvec(0), MAXITER, precision);

    return sol.total_return(initial);
}

void MDPI::to_csv(ostream& output_mdp, ostream& output_state2obs,
                  ostream& output_initial, bool headers) const{

    // save the MDP
    mdp->to_csv(output_mdp, headers);

    // save state maps
    if(headers){
        output_state2obs << "idstate,idobs" << endl;
    }
    for(auto i : indices(state2observ)){
        output_state2obs << i << "," << state2observ[i] << endl;
    }

    // save the initial distribution
    if(headers){
        output_initial << "idstate,probability" << endl;
    }
    const indvec& inindices = initial.get_indices();
    const numvec& probabilities = initial.get_probabilities();

    for(auto i : indices(inindices)){
        output_initial << inindices[i] << "," << probabilities[i] << endl;
    }
}

void MDPI::to_csv_file(const string& output_mdp, const string& output_state2obs,
                       const string& output_initial, bool headers) const{


    // open file streams
    ofstream ofs_mdp(output_mdp),
                ofs_state2obs(output_state2obs),
                ofs_initial(output_initial);

    // save the data
    to_csv(ofs_mdp, ofs_state2obs, ofs_initial, headers);

    // close streams
    ofs_mdp.close(); ofs_state2obs.close(); ofs_initial.close();
}

template<typename T>
unique_ptr<T> MDPI::from_csv(istream& input_mdp, istream& input_state2obs,
                                istream& input_initial, bool headers){


    // read mdp
    MDP mdp;
    craam::from_csv(mdp,input_mdp);

    // read state2obs
    string line;
    if(headers) input_state2obs >> line; // skip the header

    indvec state2obs(mdp.state_count());
    input_state2obs >> line;
    while(input_state2obs.good()){
        string cellstring;
        stringstream linestream(line);

        getline(linestream, cellstring, ',');
        auto idstate = stoi(cellstring);
        getline(linestream, cellstring, ',');
        auto idobs = stoi(cellstring);
        state2obs[idstate] = idobs;

        input_state2obs >> line;
    }

    // read initial distribution
    if(headers) input_initial >> line; // skip the header

    Transition initial;
    input_initial >> line;
    while(input_initial.good()){
        string cellstring;
        stringstream linestream(line);

        getline(linestream, cellstring, ',');
        auto idstate = stoi(cellstring);
        getline(linestream, cellstring, ',');
        auto prob = stof(cellstring);
        initial.add_sample(idstate, prob, 0.0);

        input_initial >> line;
    }

    shared_ptr<const MDP> csmdp = make_shared<const MDP>(std::move(mdp));
    return make_unique<T>(csmdp, state2obs, initial);
}

template
unique_ptr<MDPI> MDPI::from_csv<MDPI>(istream& input_mdp, istream& input_state2obs,
                                istream& input_initial, bool headers);

template
unique_ptr<MDPI_R> MDPI::from_csv<MDPI_R>(istream& input_mdp, istream& input_state2obs,
                                istream& input_initial, bool headers);


template<typename T>
unique_ptr<T> MDPI::from_csv_file(const string& input_mdp, const string& input_state2obs,
                                     const string& input_initial, bool headers){

    // open files
    ifstream ifs_mdp(input_mdp),
                ifs_state2obs(input_state2obs),
                ifs_initial(input_initial);

    // transfer method call
    return from_csv<T>(ifs_mdp, ifs_state2obs, ifs_initial, headers);
}

template
unique_ptr<MDPI> MDPI::from_csv_file<MDPI>(const string& input_mdp, const string& input_state2obs,
                                     const string& input_initial, bool headers);
template
unique_ptr<MDPI_R> MDPI::from_csv_file<MDPI_R>(const string& input_mdp, const string& input_state2obs,
                                     const string& input_initial, bool headers);

Transition MDPI::transition2obs(const Transition& tran){
    if((size_t) tran.max_index() >= state_count())
        throw invalid_argument("Transition to a non-existing state.");

    Transition result;

    for(auto i : range((size_t)0, tran.size())){
        const long state = tran.get_indices()[i];
        const prec_t prob = tran.get_probabilities()[i];
        const prec_t reward = tran.get_rewards()[i];

        result.add_sample(state2obs(state), prob, reward);
    }
    return result;
}

MDPI_R::MDPI_R(const shared_ptr<const MDP>& mdp, const indvec& state2observ,
            const Transition& initial) : MDPI(mdp, state2observ, initial),
            state2outcome(mdp->state_count(),-1){
    initialize_robustmdp();
}


MDPI_R::MDPI_R(const MDP& mdp, const indvec& state2observ,
            const Transition& initial) : MDPI(mdp, state2observ, initial),
            state2outcome(mdp.state_count(),-1){


    initialize_robustmdp();
}

void MDPI_R::initialize_robustmdp(){

    // Determine the number of state2observ
    auto obs_count = *max_element(state2observ.begin(), state2observ.end()) + 1;

    // keep track of the number of outcomes for each
    indvec outcome_count(obs_count, 0);

    for(size_t state_index : indices(*mdp)){
        auto obs = state2observ[state_index];

        // make sure to at least create a terminal state when there are no actions for it
        robust_mdp.create_state(obs);

        // maps the transitions
        for(auto action_index : range(0l, action_counts[obs])){
            // get original MDP transition
            const Transition& old_tran = mdp->get_state(state_index).get_action(action_index).get_outcome();
            // create a new transition
            Transition& new_tran = robust_mdp.create_state(obs).create_action(action_index).create_outcome(outcome_count[obs]);

            // copy the original transitions (they are automatically consolidated while being added)
            for(auto k : range((size_t) 0, old_tran.size())){
                new_tran.add_sample(state2observ[old_tran.get_indices()[k]],
                                    old_tran.get_probabilities()[k],
                                    old_tran.get_rewards()[k]);
            }

        }
        state2outcome[state_index] = outcome_count[obs]++;
    }
}

void MDPI_R::update_importance_weights(const numvec& weights){

    if(weights.size() != state_count()){
        throw invalid_argument("Size of distribution must match the number of states.");
    }

    // loop over all mdp states and set weights
    for(size_t i : indices(weights)){
        const auto rmdp_stateid = state2observ[i];
        const auto rmdp_outcomeid = state2outcome[i];

        // loop over all actions
        auto& rstate = robust_mdp.get_state(rmdp_stateid);
        for(size_t ai : indices(rstate)){
            rstate.get_action(ai).set_distribution(rmdp_outcomeid, weights[i]);
        }
    }

    // now normalize the weights to they sum to one
    for(size_t si : indices(robust_mdp)){
        auto& s = robust_mdp.get_state(si);
        for(size_t ai : indices(s)){
            auto& a = s.get_action(ai);
            // check if the distribution sums to 0 (not visited)
            const numvec& dist = a.get_distribution();
            if(accumulate(dist.begin(), dist.end(), 0.0) > 0.0){
                a.normalize_distribution();
            }
            else{
                // just set it to be uniform
                a.uniform_distribution();
            }
        }
    }
}

/*template<class T>
void print_vector(vector<T> vec){
    for(auto&& p : vec){
        cout << p << " ";
    }
}*/
indvec MDPI_R::solve_reweighted(long iterations, prec_t discount, const indvec& initobspol){

    // the nature policy is simply all zeros
    const indvec nature(state_count(), 0);

    if(initobspol.size() > 0 && initobspol.size() != obs_count()){
        throw invalid_argument("Initial policy must be defined for all observations.");
    }

    indvec obspol(initobspol);                   // return observation policy
    if(obspol.size() == 0){
        obspol.resize(obs_count(),0);
    }
    indvec statepol(state_count(),0);         // state policy that corresponds to the observation policy
    obspol2statepol(obspol,statepol);

    // map the initial distribution to observations in order to evaluate the return
    const Transition oinitial = transition2obs(initial);

    for(auto iter : range(0l, iterations)){
        (void) iter; // to remove the warning

        // compute state distribution
        numvec&& importanceweights = mdp->ofreq_mat(initial, discount, statepol, nature);

        // update importance weights
        update_importance_weights(importanceweights);

        // compute solution of the robust MDP with the new weights
        auto&& s = robust_mdp.mpi_jac(Uncertainty::Average, discount);

        // update the policy for the underlying states
        obspol = s.policy;

        // map the observation policy to the individual states
        obspol2statepol(obspol, statepol);

    }
    return obspol;
}

}}



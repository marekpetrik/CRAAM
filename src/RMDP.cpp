#include "RMDP.hpp"

#include <limits>
#include <algorithm>
#include <string>
#include <sstream>
#include <utility>
#include <iostream>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/lu.hpp>

#include "cpp11-range-master/range.hpp"

// this is just for a matrix printout / remove if cout is not used
// #include <boost/numeric/ublas/io.hpp>

using namespace util::lang;

namespace craam {


/// **************************************************************************************
///  Generic MDP Class
/// **************************************************************************************


template<class SType>
SType& GRMDP<SType>::create_state(long stateid) {
    assert(stateid >= 0);

    if(stateid >= (long) states.size())
        states.resize(stateid + 1);
    return states[stateid];
}

template<class SType>
bool GRMDP<SType>::is_normalized() const{
    for(auto const& s : states){
        for(auto const& a : s.get_actions()){
            for(auto const& t : a.get_outcomes()){
                if(!t.is_normalized())
                    return false;
            }
        }
    }
    return true;
}

template<class SType>
void GRMDP<SType>::normalize(){
     for(SType& s : states)
        s.normalize();
}

template<class SType>
long GRMDP<SType>::is_policy_correct(const ActionPolicy& policy,
                           const OutcomePolicy& natpolicy) const {

    for(auto si : indices(states) ){
        // ignore terminal states
        if(states[si].is_terminal())
            continue;

        // call function of the state
        if(!states[si].is_action_outcome_correct(policy[si], natpolicy[si]))
            return si;
    }
    return -1;
}

template<class SType>
void GRMDP<SType>::to_csv(ostream& output, bool header) const{

    //write header is so requested
    if(header){
        output << "idstatefrom," << "idaction," <<
            "idoutcome," << "idstateto," << "probability," << "reward" << endl;
    }

    //idstatefrom
    for(size_t i = 0l; i < this->states.size(); i++){
        const auto& actions = (this->states[i]).get_actions();
        //idaction
        for(size_t j = 0; j < actions.size(); j++){
            const auto& outcomes = actions[j].get_outcomes();
            //idoutcome
            for(size_t k = 0; k < outcomes.size(); k++){
                const auto& tran = outcomes[k];

                auto& indices = tran.get_indices();
                const auto& rewards = tran.get_rewards();
                const auto& probabilities = tran.get_probabilities();
                //idstateto
                for (size_t l = 0; l < tran.size(); l++){
                    output << i << ',' << j << ',' << k << ',' << indices[l] << ','
                            << probabilities[l] << ',' << rewards[l] << endl;
                }
            }
        }
    }
}

template<class SType>
void GRMDP<SType>::to_csv_file(const string& filename, bool header) const{
    ofstream ofs(filename, ofstream::out);

    to_csv(ofs,header);
    ofs.close();
}

template<class SType>
string GRMDP<SType>::to_string() const {
    string result;

    for(size_t si : indices(states)){
        const auto& s = get_state(si);
        result.append(std::to_string(si));
        result.append(" : ");
        result.append(std::to_string(s.action_count()));
        result.append("\n");
        for(size_t ai : indices(s)){
            result.append("    ");
            result.append(std::to_string(ai));
            result.append(" : ");
            const auto& a = s.get_action(ai);
            a.to_string(result);
            result.append("\n");
        }
    }
    return result;
}

template<class SType>
auto GRMDP<SType>::vi_gs(Uncertainty type, prec_t discount, numvec valuefunction,
                         unsigned long iterations, prec_t maxresidual) const
                            -> SolType {

    //static_assert(type != Uncertainty::Robust || type != Uncertainty::Optimistic || type != Uncertainty::Average,
    //              "Unknown/invalid (average not supported) optimization type.");


    if(valuefunction.size() > 0){
        if(valuefunction.size() != states.size())
            throw invalid_argument("Incorrect dimensions of value function.");
    }else
        valuefunction.assign(state_count(), 0.0);


    GRMDP<SType>::ActionPolicy policy(states.size());
    GRMDP<SType>::OutcomePolicy outcomes(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            const auto& state = states[s];

            tuple<ActionId,OutcomeId,prec_t> newvalue;

            switch(type){
            case Uncertainty::Robust:
                newvalue = state.max_min(valuefunction,discount);
                break;
            case Uncertainty::Optimistic:
                newvalue = state.max_max(valuefunction,discount);
                break;
            case Uncertainty::Average:
                pair<typename SType::ActionId,prec_t> avgvalue =
                    state.max_average(valuefunction,discount);
                newvalue = make_tuple(avgvalue.first,OutcomeId(),avgvalue.second);
                break;
            }

            residual = max(residual, abs(valuefunction[s] - get<2>(newvalue)));
            valuefunction[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }
    }
    return SolType(valuefunction,policy,outcomes,residual,i);
}

template<class SType>
auto GRMDP<SType>::vi_jac(Uncertainty type, prec_t discount, const numvec& valuefunction, unsigned long iterations, prec_t maxresidual) const -> SolType{

    //static_assert(type != Uncertainty::Robust || type != Uncertainty::Optimistic || type != Uncertainty::Average,
    //                      "Unknown/invalid (average not supported) optimization type.");

    if( (valuefunction.size() > 0) && (valuefunction.size() != states.size()) )
        throw invalid_argument("Incorrect size of value function.");

    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    GRMDP<SType>::ActionPolicy policy(states.size());
    GRMDP<SType>::OutcomePolicy outcomes(states.size());

    numvec residuals(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        numvec & sourcevalue = i % 2 == 0 ? oddvalue  : evenvalue;
        numvec & targetvalue = i % 2 == 0 ? evenvalue : oddvalue;

        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            const auto& state = states[s];

            tuple<ActionId,OutcomeId,prec_t> newvalue;

            switch(type){
            case Uncertainty::Robust:
                newvalue = state.max_min(sourcevalue,discount);
                break;
            case Uncertainty::Optimistic:
                newvalue = state.max_max(sourcevalue,discount);
                break;
            case Uncertainty::Average:
                pair<typename SType::ActionId,prec_t> avgvalue =
                    state.max_average(sourcevalue,discount);
                newvalue = make_tuple(avgvalue.first,OutcomeId(),avgvalue.second);
                break;
            }

            residuals[s] = abs(sourcevalue[s] - get<2>(newvalue));
            targetvalue[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }
        residual = *max_element(residuals.begin(),residuals.end());
    }
    numvec & valuenew = i % 2 == 0 ? oddvalue : evenvalue;
    return SolType(valuenew,policy,outcomes,residual,i);
}

template<class SType>
auto GRMDP<SType>::mpi_jac(Uncertainty type,
                           prec_t discount,
                           const numvec& valuefunction,
                           unsigned long iterations_pi,
                           prec_t maxresidual_pi,
                            unsigned long iterations_vi,
                            prec_t maxresidual_vi) const -> SolType{

    //static_assert(type != Uncertainty::Robust || type != Uncertainty::Optimistic || type != Uncertainty::Average,
    //                      "Unknown/invalid (average not supported) optimization type.");


    if( (valuefunction.size() > 0) && (valuefunction.size() != states.size()) )
        throw invalid_argument("Incorrect size of value function.");

    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    GRMDP<SType>::ActionPolicy policy(states.size());
    GRMDP<SType>::OutcomePolicy outcomes(states.size());

    numvec residuals(states.size());

    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    numvec * sourcevalue = & oddvalue;
    numvec * targetvalue = & evenvalue;

    for(i = 0; i < iterations_pi; i++){

        std::swap<numvec*>(targetvalue, sourcevalue);

        prec_t residual_vi = numeric_limits<prec_t>::infinity();

        // update policies
        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            const auto& state = states[s];

            tuple<ActionId,OutcomeId,prec_t> newvalue;

            switch(type){
            case Uncertainty::Robust:
                newvalue = state.max_min(*sourcevalue,discount);
                break;
            case Uncertainty::Optimistic:
                newvalue = state.max_max(*sourcevalue,discount);
                break;
            case Uncertainty::Average:
                pair<typename SType::ActionId,prec_t> avgvalue =
                    state.max_average(*sourcevalue,discount);
                newvalue = make_tuple(avgvalue.first,OutcomeId(),avgvalue.second);
                break;
            }

            residuals[s] = abs((*sourcevalue)[s] - get<2>(newvalue));
            (*targetvalue)[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }

        residual_pi = *max_element(residuals.begin(),residuals.end());

        // the residual is sufficiently small
        if(residual_pi <= maxresidual_pi)
            break;

        // compute values using value iteration
        for(size_t j = 0; j < iterations_vi && residual_vi > maxresidual_vi; j++){

            swap(targetvalue, sourcevalue);

            #pragma omp parallel for
            for(auto s = 0l; s < (long) states.size(); s++){
                prec_t newvalue;

                switch(type){
                case Uncertainty::Robust:
                case Uncertainty::Optimistic:
                    newvalue = states[s].fixed_fixed(*sourcevalue,discount,policy[s],outcomes[s]);
                    break;
                case Uncertainty::Average:
                    newvalue = states[s].fixed_average(*sourcevalue,discount,policy[s]);
                    break;
                }

                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
    }
    numvec & valuenew = *targetvalue;
    return SolType(valuenew,policy,outcomes,residual_pi,i);
}

template<class SType>
auto GRMDP<SType>::vi_jac_fix(prec_t discount,
                            const ActionPolicy& policy,
                            const OutcomePolicy& natpolicy,
                            const numvec& valuefunction,
                            unsigned long iterations,
                            prec_t maxresidual) const -> SolType{

    if(policy.size() != state_count())
        throw invalid_argument("Dimension of the policy must match the state count.");
    if(natpolicy.size() != state_count())
        throw invalid_argument("Dimension of the nature's policy must match the state count.");

    numvec oddvalue(0);        // set in even iterations (0 is even)
    numvec evenvalue(0);       // set in odd iterations

    if(valuefunction.size() > 0){
        oddvalue = valuefunction;
        evenvalue = valuefunction;
    }else{
        oddvalue.assign(states.size(),0);
        evenvalue.assign(states.size(),0);
    }

    numvec residuals(states.size());
    prec_t residual = numeric_limits<prec_t>::infinity();

    size_t j; // defined here to be able to report the number of iterations

    numvec * sourcevalue = & oddvalue;
    numvec * targetvalue = & evenvalue;

    for(j = 0; j < iterations && residual > maxresidual; j++){

        swap(targetvalue, sourcevalue);

        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            auto newvalue = states[s].fixed_fixed(*sourcevalue,discount,policy[s],natpolicy[s]);

            residuals[s] = abs((*sourcevalue)[s] - newvalue);
            (*targetvalue)[s] = newvalue;
        }
        residual = *max_element(residuals.begin(),residuals.end());
    }

    return SolType(*targetvalue,policy,natpolicy,residual,j);
}

template<class SType>
numvec GRMDP<SType>::ofreq_mat(const Transition& init, prec_t discount,
                       const ActionPolicy& policy, const OutcomePolicy& nature) const{
    const auto n = state_count();

    // initial distribution
    auto&& initial_svec = init.probabilities_vector(n);
    ublas::vector<prec_t> initial_vec(n);
    // TODO: this is a wasteful copy operation
    copy(initial_svec.begin(), initial_svec.end(), initial_vec.data().begin());

    // get transition matrix
    unique_ptr<ublas::matrix<prec_t>> t_mat(transition_mat_t(policy,nature));

    // construct main matrix
    (*t_mat) *= -discount;
    (*t_mat) += ublas::identity_matrix<prec_t>(n);

    // solve set of linear equations
    ublas::permutation_matrix<prec_t> P(n);
    ublas::lu_factorize(*t_mat,P);
    ublas::lu_substitute(*t_mat,P,initial_vec);

    // copy the solution back to a vector
    copy(initial_vec.begin(), initial_vec.end(), initial_svec.begin());

    return initial_svec;
}

template<class SType>
numvec GRMDP<SType>::rewards_state(const ActionPolicy& policy, const OutcomePolicy& nature) const{
    const auto n = state_count();
    numvec rewards(n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        const SType& state = get_state(s);
        if(state.is_terminal())
            rewards[s] = 0;
        else
            rewards[s] = state.mean_reward(policy[s], nature[s]);
    }
    return rewards;
}

template<class SType>
unique_ptr<ublas::matrix<prec_t>>
GRMDP<SType>::transition_mat(const ActionPolicy& policy, const OutcomePolicy& nature) const{

    const size_t n = state_count();
    unique_ptr<ublas::matrix<prec_t>> result(new ublas::matrix<prec_t>(n,n));
    *result = ublas::zero_matrix<prec_t>(n,n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        const Transition&& t = states[s].mean_transition(policy[s],nature[s]);
        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        for(size_t j=0; j < t.size(); j++){
            (*result)(s,indexes[j]) = probabilities[j];
        }
    }
    return result;
}

template<class SType>
unique_ptr<ublas::matrix<prec_t>>
GRMDP<SType>::transition_mat_t(const ActionPolicy& policy, const OutcomePolicy& nature) const{

    const size_t n = state_count();
    unique_ptr<ublas::matrix<prec_t>> result(new ublas::matrix<prec_t>(n,n));
    *result = ublas::zero_matrix<prec_t>(n,n);

    #pragma omp parallel for
    for(size_t s = 0; s < n; s++){
        // if this is a terminal state, then just go with zero probabilities
        if(states[s].is_terminal())  continue;

        const Transition&& t = states[s].mean_transition(policy[s],nature[s]);
        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        for(size_t j=0; j < t.size(); j++)
            (*result)(indexes[j],s) = probabilities[j];
    }
    return result;
}

/// **********************************************************************
/// *********************    TEMPLATE DECLARATIONS    ********************
/// **********************************************************************

template class GRMDP<RegularState>;
template class GRMDP<DiscreteRobustState>;
template class GRMDP<L1RobustState>;


}

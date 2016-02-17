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

prec_t Solution::total_return(const Transition& initial) const{

    if(initial.max_index() >= (long) valuefunction.size())
        throw invalid_argument("Too many indexes in the initial distribution.");

    return initial.compute_value(valuefunction);
}

size_t RMDP::state_count() const{
    return this->states.size();
}

void RMDP::assure_state_exists(long stateid){
    // re-sizing to accommodate the new state
    if(stateid >= (long) states.size())
        states.resize(stateid + 1);
}


void RMDP::add_transition(long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){

    if(fromid < 0l) throw invalid_argument("Fromid must be non-negative.");

    assure_state_exists(fromid);
    assure_state_exists(toid);

    this->states[fromid].add_action(actionid, outcomeid, toid, probability, reward);
}

void RMDP::add_transition_d(long fromid, long actionid, long toid, prec_t probability, prec_t reward){

    add_transition(fromid, actionid, 0, toid, probability, reward);
}


bool RMDP::is_normalized() const{

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
void RMDP::normalize(){
     for(State& s : states)
        s.normalize();
}


void RMDP::add_transitions(indvec const& fromids, indvec const& actionids, indvec const& outcomeids, indvec const& toids, numvec const& probs, numvec const& rews){

    auto s = fromids.size();
    if(s != outcomeids.size() || s != toids.size() || s != actionids.size() || s != probs.size() || s != rews.size())
        throw invalid_argument("sizes do not match.");

    for(auto l=0l; l <= (long) s; l++)
        this->add_transition(fromids[l],actionids[l],outcomeids[l],toids[l],probs[l],rews[l]);
}

void RMDP::set_uniform_thresholds(prec_t threshold){
    for(auto& s : this->states)
        s.set_thresholds(threshold);
}

long RMDP::assert_policy_correct(indvec policy, indvec natpolicy) const {

    for(auto si : range((size_t) 0, state_count())){
        // ignore terminal states
        if(states[si].is_terminal())
            continue;

        const auto p = policy[si];
        if(p < 0 || policy[si] >= (long) states[si].action_count())
            return si;
        const Action& a = states[si].get_action(p);

        const auto np = natpolicy[si];
        if(np < 0 || np >= (long) a.outcome_count())
            return si;
    }
    return -1;
}


unique_ptr<RMDP> RMDP::from_csv(istream& input, bool header){

    string line;

    // skip the first row if so instructed
    if(header) input >> line;

    unique_ptr<RMDP> result(new RMDP());

    input >> line;
    while(input.good()){
        string cellstring;
        stringstream linestream(line);
        long idstatefrom, idstateto, idaction, idoutcome;
        prec_t probability, reward;

        // read idstatefrom
        getline(linestream, cellstring, ',');
        idstatefrom = stoi(cellstring);
        // read idaction
        getline(linestream, cellstring, ',');
        idaction = stoi(cellstring);
        // read idoutcome
        getline(linestream, cellstring, ',');
        idoutcome = stoi(cellstring);
        // read idstateto
        getline(linestream, cellstring, ',');
        idstateto = stoi(cellstring);
        // read probability
        getline(linestream, cellstring, ',');
        probability = stof(cellstring);
        // read reward
        getline(linestream, cellstring, ',');
        reward = stof(cellstring);
        result->add_transition(idstatefrom,idaction,idoutcome,idstateto,probability,reward);

        input >> line;
    }

    return result;
}

void RMDP::to_csv(ostream& output, bool header) const{

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


string RMDP::to_string() const {
    string result;

    for(size_t i = 0; i < states.size(); i++){
        auto& s = get_state(i);
        result.append(std::to_string(i));
        result.append(" : ");
        result.append(std::to_string(s.action_count()));
        result.append("\n");
        for(size_t j = 0; j < s.action_count(); j++){
            result.append("    ");
            result.append(std::to_string(j));
            result.append(" : ");
            result.append(std::to_string(s.get_action(j).get_outcomes().size()));
            result.append(" / ");
            result.append(std::to_string(s.get_action(j).get_distribution().size()));
            result.append("\n");
        }
    }
    return result;
}

void RMDP::set_uniform_distribution(prec_t threshold){

    for(auto& s : states){
        for(auto& a : s.actions){
            auto outcomecount = a.get_outcomes().size();
            prec_t p = 1.0 / (prec_t) outcomecount;
            numvec distribution(outcomecount, p);
            a.set_distribution(distribution);
            a.set_threshold(threshold);
        }
    }
}

void RMDP::to_csv_file(const string& filename, bool header) const{
    ofstream ofs(filename, ofstream::out);

    to_csv(ofs,header);
    ofs.close();
}

unique_ptr<RMDP> RMDP::from_csv_file(const string& filename, bool header){
    ifstream ifs(filename);

    auto result = from_csv(ifs, header);
    ifs.close();

    return result;
}

template<SolutionType type>
Solution RMDP::vi_gs_gen(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    if(valuefunction.size() > 0){
        if(valuefunction.size() != states.size())
            throw invalid_argument("Incorrect dimensions of value function.");
    }else{
        valuefunction.assign(state_count(), 0.0);
    }

    indvec policy(states.size());
    indvec outcomes(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;

        for(size_t s = 0l; s < states.size(); s++){
            const auto& state = states[s];

            pair<long,prec_t> avgvalue;
            tuple<long,long,prec_t> newvalue;

            switch(type){
            case SolutionType::Robust:
                newvalue = state.max_min(valuefunction,discount);
                break;
            case SolutionType::Optimistic:
                newvalue = state.max_max(valuefunction,discount);
                break;
            case SolutionType::Average:
                avgvalue = state.max_average(valuefunction,discount);
                newvalue = make_tuple(avgvalue.first,-1,avgvalue.second);
                break;
            }

            residual = max(residual, abs(valuefunction[s] - get<2>(newvalue)));
            valuefunction[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }
    }
    return Solution(valuefunction,policy,outcomes,residual,i);
}

template Solution RMDP::vi_gs_gen<SolutionType::Robust>(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_gs_gen<SolutionType::Optimistic>(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_gs_gen<SolutionType::Average>(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

template<SolutionType type, NatureConstr nature>
Solution RMDP::vi_gs_cst(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    if(valuefunction.size() > 0){
        if(valuefunction.size() != states.size())
            throw invalid_argument("incorrect size of value function");
    }else{
        valuefunction.assign(state_count(), 0);
    }

    indvec policy(states.size());
    vector<numvec> outcome_dists(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        residual = 0;
        for(auto s=0l; s < (long) state_count(); s++){
            const auto& state = states[s];

            tuple<long,numvec,prec_t> newvalue;
            switch(type){
            case SolutionType::Robust:
                newvalue = state.max_min_cst<nature>(valuefunction, discount);
                break;
            case SolutionType::Optimistic:
                newvalue = state.max_max_cst<nature>(valuefunction, discount);
                break;
            default:
                static_assert(type != SolutionType::Robust || type != SolutionType::Optimistic, "Unknown/invalid (average not supported) optimization type.");
                throw invalid_argument("Unknown/invalid (average not supported) optimization type.");
            }
            residual = max(residual, abs(valuefunction[s] - get<2>(newvalue) ));
            valuefunction[s] = get<2>(newvalue);
            outcome_dists[s] = get<1>(newvalue);
            policy[s] = get<0>(newvalue);
        }
    }
    return Solution(valuefunction,policy,outcome_dists,residual,i);
}

template Solution RMDP::vi_gs_cst<SolutionType::Robust,worstcase_l1>(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_gs_cst<SolutionType::Optimistic,worstcase_l1>(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_gs_cst<SolutionType::Average,worstcase_l1>(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

template<SolutionType type>
Solution RMDP::vi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

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

    indvec policy(states.size());
    indvec outcomes(states.size());

    numvec residuals(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        numvec & sourcevalue = i % 2 == 0 ? oddvalue  : evenvalue;
        numvec & targetvalue = i % 2 == 0 ? evenvalue : oddvalue;

        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            const auto& state = states[s];

            pair<long,prec_t> avgvalue;
            tuple<long,long,prec_t> newvalue;

            switch(type){
            case SolutionType::Robust:
                newvalue = state.max_min(sourcevalue,discount);
                break;
            case SolutionType::Optimistic:
                newvalue = state.max_max(sourcevalue,discount);
                break;
            case SolutionType::Average:
                avgvalue = state.max_average(sourcevalue,discount);
                newvalue = make_tuple(avgvalue.first,-1,avgvalue.second);
                break;
            default:
                throw invalid_argument("Unknown optimization type.");
            }

            residuals[s] = abs(sourcevalue[s] - get<2>(newvalue));
            targetvalue[s] = get<2>(newvalue);

            policy[s] = get<0>(newvalue);
            outcomes[s] = get<1>(newvalue);
        }
        residual = *max_element(residuals.begin(),residuals.end());
    }
    numvec & valuenew = i % 2 == 0 ? oddvalue : evenvalue;
    return Solution(valuenew,policy,outcomes,residual,i);
}

template Solution RMDP::vi_jac_gen<SolutionType::Robust>(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_jac_gen<SolutionType::Optimistic>(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_jac_gen<SolutionType::Average>(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

template<SolutionType type,NatureConstr nature>
Solution
RMDP::vi_jac_cst(numvec const& valuefunction, prec_t discount,
                 unsigned long iterations, prec_t maxresidual) const{

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

    indvec policy(states.size());
    vector<numvec> outcome_dists(states.size());

    numvec residuals(states.size());

    prec_t residual = numeric_limits<prec_t>::infinity();
    size_t i;

    for(i = 0; i < iterations && residual > maxresidual; i++){
        numvec & sourcevalue = i % 2 == 0 ? oddvalue  : evenvalue;
        numvec & targetvalue = i % 2 == 0 ? evenvalue : oddvalue;

        #pragma omp parallel for
        for(auto s = 0l; s <  (long)this->states.size(); s++){
            const auto& state = this->states[s];

            tuple<long,numvec,prec_t> newvalue;
            switch(type){
            case SolutionType::Robust:
                newvalue = state.max_min_l1(sourcevalue,discount);
                break;
            case SolutionType::Optimistic:
                newvalue = state.max_max_l1(sourcevalue,discount);
                break;
            default:
                static_assert(type != SolutionType::Robust || type != SolutionType::Optimistic, "Unknown/invalid (average not supported) optimization type.");
                throw invalid_argument("Unknown/invalid (average not supported) optimization type.");
            }

            residuals[s] = abs(sourcevalue[s] - get<2>(newvalue));
            targetvalue[s] = get<2>(newvalue);
            outcome_dists[s] = get<1>(newvalue);
            policy[s] = get<0>(newvalue);

        }
        residual = *max_element(residuals.begin(),residuals.end());
    }
    numvec & valuenew = i % 2 == 0 ? oddvalue : evenvalue;
    return Solution(valuenew,policy,outcome_dists,residual,i);
}

template Solution RMDP::vi_jac_cst<SolutionType::Robust,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_jac_cst<SolutionType::Optimistic,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;
template Solution RMDP::vi_jac_cst<SolutionType::Average,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const;

template<SolutionType type>
Solution
RMDP::mpi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                  unsigned long iterations_vi, prec_t maxresidual_vi) const{

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

    indvec policy(states.size());
    indvec outcomes(states.size());

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

            pair<long,prec_t> avgvalue;
            tuple<long,long,prec_t> newvalue;

            switch(type){
            case SolutionType::Robust:
                newvalue = state.max_min(*sourcevalue,discount);
                break;
            case SolutionType::Optimistic:
                newvalue = state.max_max(*sourcevalue,discount);
                break;
            case SolutionType::Average:
                avgvalue = state.max_average(*sourcevalue,discount);
                newvalue = make_tuple(avgvalue.first,-1,avgvalue.second);
                break;
            default:
                throw invalid_argument("unknown optimization type.");
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
                case SolutionType::Robust:
                case SolutionType::Optimistic:
                    newvalue = states[s].fixed_fixed(*sourcevalue,discount,policy[s],outcomes[s]);
                    break;
                case SolutionType::Average:
                    newvalue = states[s].fixed_average(*sourcevalue,discount,policy[s]);
                    break;
                default:
                    throw invalid_argument("Unknown optimization type.");
                }
                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
    }
    numvec & valuenew = *targetvalue;
    return Solution(valuenew,policy,outcomes,residual_pi,i);
}

template Solution RMDP::mpi_jac_gen<SolutionType::Robust>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
template Solution RMDP::mpi_jac_gen<SolutionType::Optimistic>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
template Solution RMDP::mpi_jac_gen<SolutionType::Average>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;

template<SolutionType type, NatureConstr nature>
Solution RMDP::mpi_jac_cst(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{


    if(type == SolutionType::Average) throw invalid_argument("computing average is not supported by this function");

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

    indvec policy(states.size());

    vector<numvec> outcomes(states.size());

    numvec residuals(states.size());

    prec_t residual_pi = numeric_limits<prec_t>::infinity();

    size_t i; // defined here to be able to report the number of iterations

    numvec * sourcevalue = & oddvalue;
    numvec * targetvalue = & evenvalue;

    for(i = 0; i < iterations_pi; i++){

        std::swap(targetvalue, sourcevalue);

        prec_t residual_vi = numeric_limits<prec_t>::infinity();

        // update policies
        #pragma omp parallel for
        for(auto s = 0l; s < (long) states.size(); s++){
            const auto& state = states[s];

            // TODO: change to an rvalue?
            tuple<long,numvec,prec_t> newvalue;

            switch(type){
            case SolutionType::Robust:
                newvalue = state.max_min_l1(*sourcevalue,discount);
                break;
            case SolutionType::Optimistic:
                newvalue = state.max_max_l1(*sourcevalue,discount);
                break;
            default:
                throw invalid_argument("unsupported optimization type.");
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
                auto newvalue = states[s].fixed_average(*sourcevalue,discount,policy[s],outcomes[s]);

                residuals[s] = abs((*sourcevalue)[s] - newvalue);
                (*targetvalue)[s] = newvalue;
            }
            residual_vi = *max_element(residuals.begin(),residuals.end());
        }
    }

    numvec & valuenew = *targetvalue;

    return Solution(valuenew,policy,outcomes,residual_pi,i);
}

Solution RMDP::vi_jac_fix(const numvec& valuefunction, prec_t discount, const indvec& policy,
                          const indvec& natpolicy, unsigned long iterations,
                          prec_t maxresidual) const{

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

    return Solution(*targetvalue,policy,natpolicy,residual,j);
}

Solution RMDP::vi_jac_fix_ave(const numvec& valuefunction, prec_t discount, const indvec& policy,
                              unsigned long iterations,
                              prec_t maxresidual) const{

    if(policy.size() != state_count())
        throw invalid_argument("Dimension of the policy must match the state count.");

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
            auto newvalue = states[s].fixed_average(*sourcevalue,discount,policy[s]);

            residuals[s] = abs((*sourcevalue)[s] - newvalue);
            (*targetvalue)[s] = newvalue;
        }
        residual = *max_element(residuals.begin(),residuals.end());
    }

    return Solution(*targetvalue,policy,indvec(0),residual,j);
}

template Solution RMDP::mpi_jac_cst<SolutionType::Robust,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
template Solution RMDP::mpi_jac_cst<SolutionType::Optimistic,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
template Solution RMDP::mpi_jac_cst<SolutionType::Average,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;

Solution RMDP::vi_gs_rob(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    return vi_gs_gen<SolutionType::Robust>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_opt(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    return vi_gs_gen<SolutionType::Optimistic>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_ave(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    return vi_gs_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_l1_rob(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    return vi_gs_cst<SolutionType::Robust, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_l1_opt(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    return vi_gs_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

     return vi_jac_gen<SolutionType::Robust>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

     return vi_jac_gen<SolutionType::Optimistic>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_ave(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

     return vi_jac_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
}


Solution RMDP::vi_jac_l1_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

     return vi_jac_cst<SolutionType::Robust, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_l1_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{

    return vi_jac_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}


// modified policy iteration
Solution RMDP::mpi_jac_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

     return mpi_jac_gen<SolutionType::Robust>(valuefunction, discount, iterations_pi, maxresidual_pi,
                 iterations_vi, maxresidual_vi);

}

Solution RMDP::mpi_jac_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

     return mpi_jac_gen<SolutionType::Optimistic>(valuefunction, discount, iterations_pi, maxresidual_pi,
                 iterations_vi, maxresidual_vi);

}

Solution RMDP::mpi_jac_ave(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

     return mpi_jac_gen<SolutionType::Average>(valuefunction, discount, iterations_pi, maxresidual_pi,
                 iterations_vi, maxresidual_vi);
}


Solution RMDP::mpi_jac_l1_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

     return mpi_jac_cst<SolutionType::Robust,worstcase_l1>(valuefunction, discount, iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi);
}

Solution RMDP::mpi_jac_l1_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

     return mpi_jac_cst<SolutionType::Optimistic,worstcase_l1>(valuefunction, discount, iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi);
}


numvec RMDP::ofreq_mat(const Transition& init, prec_t discount, const indvec& policy, const indvec& nature) const{
    const auto n = state_count();

    // initial distribution
    auto&& initial_svec = init.probabilities_vector(n);
    ublas::vector<prec_t> initial_vec(n);
    copy(initial_svec.begin(), initial_svec.end(), initial_vec.data().begin());

    // get transition matrix
    unique_ptr<ublas::matrix<prec_t>> t_mat(transition_mat_t(policy,nature));

    //cout << "Transition matrix transpose P^T: " << *t_mat << endl;

    // construct main matrix
    (*t_mat) *= -discount;
    (*t_mat) += ublas::identity_matrix<prec_t>(n);

    //cout << "Constructed matrix (A): " << *t_mat << endl;

    // solve set of linear equations
    ublas::permutation_matrix<prec_t> P(n);
    ublas::lu_factorize(*t_mat,P);
    ublas::lu_substitute(*t_mat,P,initial_vec);

    // copy the solution back to a vector
    copy(initial_vec.begin(), initial_vec.end(), initial_svec.begin());

    return initial_svec;
}

numvec RMDP::rewards_state(const indvec& policy, const indvec& nature) const{
    const auto n = state_count();
    numvec rewards(n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        const State& state = get_state(s);
        if(state.is_terminal())
            rewards[s] = 0;
        else
            rewards[s] = state.get_transition(policy[s],nature[s]).mean_reward();
    }
    return rewards;
}

unique_ptr<ublas::matrix<prec_t>> RMDP::transition_mat(const indvec& policy, const indvec& nature) const{
    const size_t n = state_count();
    unique_ptr<ublas::matrix<prec_t>> result(new ublas::matrix<prec_t>(n,n));
    *result = ublas::zero_matrix<prec_t>(n,n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        const Transition& t = get_transition(s,policy[s],nature[s]);
        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        for(size_t j=0; j < t.size(); j++){
            (*result)(s,indexes[j]) = probabilities[j];
        }
    }
    return result;
}

unique_ptr<ublas::matrix<prec_t>> RMDP::transition_mat_t(const indvec& policy, const indvec& nature) const{
    const size_t n = state_count();
    unique_ptr<ublas::matrix<prec_t>> result(new ublas::matrix<prec_t>(n,n));
    *result = ublas::zero_matrix<prec_t>(n,n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        // if this is a terminal state, then just go with zero probabilities
        if(states[s].is_terminal())  continue;

        const Transition& t = get_transition(s,policy[s],nature[s]);
        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        for(size_t j=0; j < t.size(); j++){
            (*result)(indexes[j],s) = probabilities[j];
            //cout << indexes[j] << " " << s << " " << probabilities[j] << endl;
        }
    }
    //cout << *result << endl;
    return result;
}

Transition& RMDP::create_transition(long fromid, long actionid, long outcomeid){
    if(fromid < 0l) throw invalid_argument("Fromid must be non-negative.");

    assure_state_exists(fromid);

    return this->states[fromid].create_transition(actionid, outcomeid);
}

Transition& RMDP::get_transition(long stateid, long actionid, long outcomeid){
    if(stateid < 0l || stateid >= (long) this->states.size()){
        throw invalid_argument("Invalid state number");
    }
    return states[stateid].get_transition(actionid,outcomeid);
}


const Transition& RMDP::get_transition(long stateid, long actionid, long outcomeid) const{
    if(stateid < 0l || stateid >= (long) this->states.size()){
        throw invalid_argument("Invalid state number");
    }
    return states[stateid].get_transition(actionid,outcomeid);
}


}

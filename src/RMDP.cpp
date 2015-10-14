#include "RMDP.hpp"

#include <limits>
#include <algorithm>
#include <string>
#include <sstream>
#include <utility>
#include <iostream>

namespace craam {

prec_t Solution::total_return(const Transition& initial) const{
    /**
        Computes the total return of the solution given the initial
        distribution.

        \param initial The initial distribution
     */
    if(initial.max_index() >= (long) valuefunction.size())
        throw invalid_argument("Too many indexes in the initial distribution.");

    return initial.compute_value(valuefunction);
}

size_t RMDP::state_count() const{
    return this->states.size();
}


void RMDP::add_transition(long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward){
    /**
       Adds a transition probability

       \param fromid Starting state ID
       \param actionid Action ID
       \param outcomeid Outcome ID (A single outcome corresponds to a regular MDP)
       \param toid Destination ID
       \param probability Probability of the transition (must be non-negative)
       \param reward The reward associated with the transition.
     */

    if(fromid < 0l) throw invalid_argument("Fromid must be non-negative.");

    auto newid = max(fromid,toid);
    if(newid >= (long) this->states.size()){
        // re-sizing to accommodate the new state
        (this->states).resize(newid+1);
    }

    this->states[fromid].add_action(actionid, outcomeid, toid, probability, reward);
}

void RMDP::add_transition_d(long fromid, long actionid, long toid, prec_t probability, prec_t reward){
    /** Adds a non-robust transition.  */
    add_transition(fromid, actionid, 0, toid, probability, reward);
}


bool RMDP::is_normalized() const{
    /**
       Check if all transitions in the process sum to one.

       Note that if there are no actions, or no outcomes for a state,
       the RMDP still may be normalized.

       \return True if and only if all transitions are normalized.
     */

    for(auto const& s : states){
        for(auto const& a : s.actions){
            for(auto const& t : a.outcomes){
                if(!t.is_normalized())
                    return false;
            }
        }
    }
    return true;
}
void RMDP::normalize(){
    /**
       Normalize all transitions to sum to one for all states, actions, outcomes.
     */

     for(auto& s : states){
        for(auto& a : s.actions){
            for(auto& t : a.outcomes){
                t.normalize();
            }
        }
    }
}


void RMDP::add_transitions(indvec const& fromids, indvec const& actionids, indvec const& outcomeids, indvec const& toids, numvec const& probs, numvec const& rews){
    /**
        Add multiple samples (transitions) to the MDP definition

       \param fromids Starting state ids
       \param outcomeis IDs used of the outcomes
       \param toids Destination state ids
       \param actionids
       \param probs Probabilities of the transitions
       \param rews Rewards of the transitions
     */

    auto s = fromids.size();
    if(s != outcomeids.size() || s != toids.size() || s != actionids.size() || s != probs.size() || s != rews.size())
        throw invalid_argument("sizes do not match.");

    for(auto l=0l; l <= (long) s; l++)
        this->add_transition(fromids[l],actionids[l],outcomeids[l],toids[l],probs[l],rews[l]);
}

void RMDP::set_uniform_thresholds(prec_t threshold){
    /**
       Sets thresholds for all states uniformly
     */
    for(auto& s : this->states){
        s.set_thresholds(threshold);
    }
}


unique_ptr<RMDP> RMDP::transitions_from_csv(istream& input, bool header){
    /**
       Loads an RMDP definition from a simple csv file.

       States, actions, and outcomes are identified by 0-based ids.

       The columns are separated by commas, and rows by new lines.

       The file is formatted with the following columns:
       idstatefrom, idaction, idoutcome, idstateto, probability, reward

       Note that outcome distributions are not restored.

       \param input Source of the RMDP
       \param header Whether the first line of the file represents the header.
                        The column names are not checked for correctness or number!
     */

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

void RMDP::transitions_to_csv(ostream& output, bool header) const{
    /**
       Saves the model to a stream as a simple csv file

       States, actions, and outcomes are identified by 0-based ids.

       The columns are separated by commas, and rows by new lines.

       The file is formatted with the following columns:
       idstatefrom, idaction, idoutcome, idstateto, probability, reward

       Exported and imported MDP will be be slightly different. Since action/transitions
       will not be exported if there are no actions for the state. However, when
       there is data for action 1 and action 3, action 2 will be created with no outcomes.

       Note that outcome distributions are not saved.

       \param output Output for the stream
       \param header Whether the header should be written as the
              first line of the file represents the header.
     */

    //write header is so requested
    if(header){
        output << "idstatefrom," << "idaction," <<
            "idoutcome," << "idstateto," << "probability," << "reward" << endl;
    }

    //idstatefrom
    for(size_t i = 0l; i < this->states.size(); i++){
        const auto& actions = (this->states[i]).actions;

        //idaction
        for(size_t j = 0; j < actions.size(); j++){
            const auto& outcomes = actions[j].outcomes;

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
    /** Returns a brief string representation of the MDP.

       This method is mostly suitable for analyzing small MDPs.
     */
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
            result.append(std::to_string(s.get_action(j).outcomes.size()));
            result.append(" / ");
            result.append(std::to_string(s.get_action(j).get_distribution().size()));
            result.append("\n");
        }
    }
    return result;
}

void RMDP::set_uniform_distribution(prec_t threshold){
    /**
       Sets the distribution for outcomes for each state and
       action to be uniform. It also sets the threshold to be the same
       for all states.
     */
    for(auto& s : states){
        for(auto& a : s.actions){
            auto outcomecount = a.outcomes.size();
            prec_t p = 1.0 / (prec_t) outcomecount;
            numvec distribution(outcomecount, p);
            a.set_distribution(distribution);
            a.set_threshold(threshold);
        }
    }
}

void RMDP::transitions_to_csv_file(const string& filename, bool header ) const{
    /**
       Saves the transition probabilities and rewards to a CSV file

       \param filename Name of the file
       \param header Whether to create a header of the file too
     */
    ofstream ofs;
    ofs.open(filename);

    transitions_to_csv(ofs,header);
    ofs.close();
}

template<SolutionType type>
Solution RMDP::vi_gs_gen(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Gauss-Seidel value iteration variant (not parallelized). This is a generic function,
       which can compute any solution type (robust, optimistic, or average).

       This function is suitable for computing the value function of a finite state MDP. If
       the states are ordered correctly, one iteration is enough to compute the optimal value function.
       Since the value function is updated from the first state to the last, the states should be ordered
       in reverse temporal order.

       Because this function updates the array value during the iteration, it may be
       difficult to parallelize.

       \param valuefunction Initial value function. Passed by value,
                            because it is modified. If it has size 0, then it is assumed
                            to be all 0s.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

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
    /**
       Gauss-Seidel value iteration variant with constrained nature(not parallelized).
       The natures policy is constrained, given by the function nature.

       Because this function updates the array value during the iteration, it may be
       difficult to parallelize.

       This is a generic version, which works for best/worst-case optimization and
       arbitrary constraints on nature (given by the function nature). Average case constrained
       nature is not supported.

       \param valuefunction Initial value function. Passed by value, because it is modified. When
                               it has zero length, it is assumed to be all zeros.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */
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
    /**
       Jacobi value iteration variant. The behavior of the nature depends on the values
       of parameter type. This method uses OpenMP to parallelize the computation.

       \param valuefunction Initial value function, if size zero, then considered to be all zeros.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */
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

    numvec residuals(valuefunction.size());

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
Solution RMDP::vi_jac_cst(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Jacobi value iteration variant with constrained nature. The outcomes are
       selected using nature function.

       This method uses OpenMP to parallelize the computation.

       This is a generic version, which works for best/worst-case optimization and
       arbitrary constraints on nature (given by the function nature). Average case constrained
       nature is not supported.

       \param valuefunction Initial value function.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

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

    numvec residuals(valuefunction.size());

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
Solution RMDP::mpi_jac_gen(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

    /**
       Modified policy iteration using Jacobi value iteration in the inner loop.
        The template parameter type determines the behavior of nature.

       This method generalizes modified policy iteration to robust MDPs.
       In the value iteration step, both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function, use a vector of length 0 if the value is not provided
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

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

    numvec residuals(valuefunction.size());

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
    /**
       Modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
       The template determines the constraints (by the parameter nature) and the type
       of nature (by the parameter type)

       This method generalized modified policy iteration to the robust MDP. In the value iteration step,
       both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

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

    numvec residuals(valuefunction.size());

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

template Solution RMDP::mpi_jac_cst<SolutionType::Robust,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
template Solution RMDP::mpi_jac_cst<SolutionType::Optimistic,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;
template Solution RMDP::mpi_jac_cst<SolutionType::Average,worstcase_l1>(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi, unsigned long iterations_vi, prec_t maxresidual_vi) const;


Solution RMDP::vi_gs_rob(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Gauss-Seidel value iteration variant (not parallelized). The outcomes are
       selected using worst-case nature.

       This function is suitable for computing the value function of a finite state MDP. If
       the states are ordered correctly, one iteration is enough to compute the optimal value function.
       Since the value function is updated from the first state to the last, the states should be ordered
       in reverse temporal order.

       Because this function updates the array value during the iteration, it may be
       difficult to parallelize easily.

       \param valuefunction Initial value function. Passed by value, because it is modified.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

    return vi_gs_gen<SolutionType::Robust>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_opt(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Gauss-Seidel value iteration variant (not parallelized). The outcomes are
       selected using best-case nature.

       This function is suitable for computing the value function of a finite state MDP. If
       the states are ordered correctly, one iteration is enough to compute the optimal value function.
       Since the value function is updated from the first state to the last, the states should be ordered
       in reverse temporal order.

       Because this function updates the array value during the iteration, it may be
       difficult to parallelize easily.

       \param valuefunction Initial value function. Passed by value, because it is modified.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

    return vi_gs_gen<SolutionType::Optimistic>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_ave(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Gauss-Seidel value iteration variant (not parallelized). The outcomes are
       selected using average-case nature.

       This function is suitable for computing the value function of a finite state MDP. If
       the states are ordered correctly, one iteration is enough to compute the optimal value function.
       Since the value function is updated from the first state to the last, the states should be ordered
       in reverse temporal order.

       Because this function updates the array value during the iteration, it may be
       difficult to paralelize easily.

       \param valuefunction Initial value function. Passed by value, because it is modified.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

    return vi_gs_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_l1_rob(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Robust Gauss-Seidel value iteration variant (not parallelized). The natures policy is
       constrained using L1 constraints and is worst-case.

       This function is suitable for computing the value function of a finite state MDP. If
       the states are ordered correctly, one iteration is enough to compute the optimal value function.
       Since the value function is updated from the first state to the last, the states should be ordered
       in reverse temporal order.

       Because this function updates the array value during the iteration, it may be
       difficult to parallelize.

       \param valuefunction Initial value function. Passed by value, because it is modified.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

    return vi_gs_cst<SolutionType::Robust, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_gs_l1_opt(numvec valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Optimistic Gauss-Seidel value iteration variant (not parallelized). The natures policy is
       constrained using L1 constraints and is best-case.

       This function is suitable for computing the value function of a finite state MDP. If
       the states are ordered correctly, one iteration is enough to compute the optimal value function.
       Since the value function is updated from the first state to the last, the states should be ordered
       in reverse temporal order.

       Because this function updates the array value during the iteration, it may be
       difficult to parallelize.

       This is a generic version, which works for best/worst-case optimization and
       arbitrary constraints on nature (given by the function nature). Average case constrained
       nature is not supported.

       \param valuefunction Initial value function. Passed by value, because it is modified.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

    return vi_gs_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Robust Jacobi value iteration variant. The nature behaves as worst-case.
       This method uses OpenMP to parallelize the computation.

       \param valuefunction Initial value function.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */
     return vi_jac_gen<SolutionType::Robust>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Optimistic Jacobi value iteration variant. The nature behaves as best-case.
       This method uses OpenMP to parallelize the computation.

       \param valuefunction Initial value function.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */
     return vi_jac_gen<SolutionType::Optimistic>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_ave(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Average Jacobi value iteration variant. The nature behaves as average-case.
       This method uses OpenMP to parallelize the computation.

       \param valuefunction Initial value function.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */
     return vi_jac_gen<SolutionType::Average>(valuefunction, discount, iterations, maxresidual);
}


Solution RMDP::vi_jac_l1_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Robust Jacobi value iteration variant with constrained nature. The nature is constrained
       by an L1 norm.

       This method uses OpenMP to parallelize the computation.

       \param valuefunction Initial value function.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

     return vi_jac_cst<SolutionType::Robust, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}

Solution RMDP::vi_jac_l1_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations, prec_t maxresidual) const{
    /**
       Optimistic Jacobi value iteration variant with constrained nature. The nature is constrained
       by an L1 norm.

       This method uses OpenMP to parallelize the computation.

       \param valuefunction Initial value function.
       \param discount Discount factor.
       \param iterations Maximal number of iterations to run
       \param maxresidual Stop when the maximal residual falls below this value.
     */

     return vi_jac_cst<SolutionType::Optimistic, worstcase_l1>(valuefunction, discount, iterations, maxresidual);
}


// modified policy iteration
Solution RMDP::mpi_jac_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

    /**
       Robust modified policy iteration using Jacobi value iteration in the inner loop.
       The nature behaves as worst-case.

       This method generalizes modified policy iteration to robust MDPs.
       In the value iteration step, both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

     return mpi_jac_gen<SolutionType::Robust>(valuefunction, discount, iterations_pi, maxresidual_pi,
                 iterations_vi, maxresidual_vi);

}

Solution RMDP::mpi_jac_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

    /**
       Optimistic modified policy iteration using Jacobi value iteration in the inner loop.
       The nature behaves as best-case.

       This method generalizes modified policy iteration to robust MDPs.
       In the value iteration step, both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

     return mpi_jac_gen<SolutionType::Optimistic>(valuefunction, discount, iterations_pi, maxresidual_pi,
                 iterations_vi, maxresidual_vi);

}

Solution RMDP::mpi_jac_ave(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{

    /**
       Average modified policy iteration using Jacobi value iteration in the inner loop.
       The nature behaves as average-case.

       This method generalizes modified policy iteration to robust MDPs.
       In the value iteration step, both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

     return mpi_jac_gen<SolutionType::Average>(valuefunction, discount, iterations_pi, maxresidual_pi,
                 iterations_vi, maxresidual_vi);
}


Solution RMDP::mpi_jac_l1_rob(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{
    /**
       Robust modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
       The constraints are defined by the L1 norm and the nature is worst-case.

       This method generalized modified policy iteration to the robust MDP. In the value iteration step,
       both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

     return mpi_jac_cst<SolutionType::Robust,worstcase_l1>(valuefunction, discount, iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi);
}

Solution RMDP::mpi_jac_l1_opt(numvec const& valuefunction, prec_t discount, unsigned long iterations_pi, prec_t maxresidual_pi,
                 unsigned long iterations_vi, prec_t maxresidual_vi) const{
    /**
       Optimistic modified policy iteration using Jacobi value iteration in the inner loop and constrained nature.
       The constraints are defined by the L1 norm and the nature is best-case.

       This method generalized modified policy iteration to the robust MDP. In the value iteration step,
       both the action *and* the outcome are fixed.

       Note that the total number of iterations will be bounded by iterations_pi * iterations_vi

       \param valuefunction Initial value function
       \param discount Discount factor
       \param iterations_pi Maximal number of policy iteration steps
       \param maxresidual_pi Stop the outer policy iteration when the residual drops below this threshold.
       \param iterations_vi Maximal number of inner loop value iterations
       \param maxresidual_vi Stop the inner policy iteration when the residual drops below this threshold.
                    This value should be smaller than maxresidual_pi

       \return Computed (approximate) solution
     */

     return mpi_jac_cst<SolutionType::Optimistic,worstcase_l1>(valuefunction, discount, iterations_pi, maxresidual_pi, iterations_vi, maxresidual_vi);
}


numvec RMDP::ofreq_mat(const Transition& init, prec_t discount, const indvec& policy, const indvec& nature) const{
    /**
        Computes occupancy frequencies using matrix representation

        \param init Initial distribution (alpha)
        \param discount Discount factor (gamma)
        \param policy Policy of the decision maker
        \param nature Policy of nature
     */

    // initialize
    // TODO: the copy here could be easily eliminated, is it worth it?
    const auto&& initial_d = arma::vec(init.probabilities_vector(state_count()));

    unique_ptr<arma::SpMat<prec_t>> t_mat(transition_mat_t(policy,nature));

    (*t_mat) *= -discount;
    (*t_mat) += arma::speye(state_count(),state_count());

    const auto&& frequency = arma::spsolve(*t_mat,initial_d);

    return arma::conv_to<numvec>::from(frequency);
}

numvec RMDP::rewards_state(const indvec& policy, const indvec& nature) const{
    /**
        Constructs the rewards vector for each state for the RMDP.

        \param policy Policy of the decision maker
        \param nature Policy of nature
     */

    const auto n = state_count();
    numvec rewards(n);

    #pragma omp parallel for
    for(size_t s=0; s < n; s++){
        rewards[s] = get_transition(s,policy[s],nature[s]).mean_reward();
    }
    return rewards;
}

unique_ptr<arma::SpMat<prec_t>> RMDP::transition_mat(const indvec& policy, const indvec& nature) const{
    /**
         Constructs the transition matrix for the policy.

        \param policy Policy of the decision maker
        \param nature Policy of nature
     */

    const size_t n = state_count();
    unique_ptr<arma::SpMat<prec_t>> result(new arma::SpMat<prec_t>(n,n));

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

unique_ptr<arma::SpMat<prec_t>> RMDP::transition_mat_t(const indvec& policy, const indvec& nature) const{
    /**
         Constructs a transpose of the transition matrix for the policy.

        \param policy Policy of the decision maker
        \param nature Policy of nature
     */

    const size_t n = state_count();
    unique_ptr<arma::SpMat<prec_t>> result(new arma::SpMat<prec_t>(n,n));

    for(size_t s=0; s < n; s++){
        const Transition& t = get_transition(s,policy[s],nature[s]);
        const auto& indexes = t.get_indices();
        const auto& probabilities = t.get_probabilities();

        for(size_t j=0; j < t.size(); j++){
            (*result)(indexes[j],s) = probabilities[j];
        }
    }
    return result;
}

Transition& RMDP::get_transition(long fromid, long actionid, long outcomeid){
    /**
       Return a transition for state, action, and outcome. It is created if
       necessary.
     */

    if(fromid < 0l) throw invalid_argument("Fromid must be non-negative.");

    if(fromid >= (long) this->states.size()){
        (this->states).resize(fromid+1);
    }

    return this->states[fromid].get_transition(actionid, outcomeid);
}

const Transition& RMDP::get_transition(long stateid, long actionid, long outcomeid) const{
    /**
       Returns the transition. The transition must exist.
     */
    if(stateid < 0l || stateid >= (long) this->states.size()){
        throw invalid_argument("Invalid state number");
    }
    return states[stateid].get_transition(actionid,outcomeid);
}


}

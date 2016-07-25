# distutils: language = c++
# distutils: libraries = craam
# distutils: library_dirs = ../lib 
# distutils: include_dirs = ../include 

import numpy as np 
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.memory cimport unique_ptr, shared_ptr, make_shared
from libcpp cimport bool
import statistics
from collections import namedtuple 
from math import sqrt
import warnings 
from cython.operator import dereference

cdef extern from "../include/RMDP.hpp" namespace 'craam' nogil:
                                            
    ctypedef double prec_t
    ctypedef vector[double] numvec
    ctypedef vector[long] indvec
    ctypedef unsigned long size_t
                                            
    cdef cppclass Uncertainty:
        pass 

    cdef cppclass SolutionDscDsc:
        numvec valuefunction
        indvec policy
        indvec outcomes
        prec_t residual
        long iterations

    cdef cppclass CTransition "craam::Transition":
        CTransition() 
        CTransition(const indvec& indices, const numvec& probabilities, const numvec& rewards)
        CTransition(const numvec& probabilities)

        void set_reward(long sampleid, double reward) except +
        double get_reward(long sampleid) except +

        numvec probabilities_vector(unsigned long size) 

        indvec& get_indices() 
        numvec& get_probabilities()
        numvec& get_rewards() 
        size_t size() 

    cdef cppclass CRegularAction "craam::RegularAction":
        CTransition& get_outcome(long outcomeid)
        size_t outcome_count()

    cdef cppclass CRegularState "craam::RegularState":
        CRegularAction& get_action(long actionid)
        size_t action_count()

    cdef cppclass CMDP "craam::MDP":
        CMDP(long)
        CMDP(const CMDP&)
        CMDP()

        size_t state_count() 
        CRegularState& get_state(long stateid)

        SolutionDscDsc vi_jac(Uncertainty uncert, prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual) const;

        SolutionDscDsc vi_gs(Uncertainty uncert, prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual) const;

        SolutionDscDsc mpi_jac(Uncertainty uncert,
                        prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations_pi,
                        prec_t maxresidual_pi,
                        unsigned long iterations_vi,
                        prec_t maxresidual_vi) const;

        SolutionDscDsc vi_jac_fix(prec_t discount,
                        const indvec& policy,
                        const indvec& natpolicy,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual=SOLPREC) const;
        
        string to_json() const;


cdef extern from "../include/RMDP.hpp" namespace 'craam::Uncertainty' nogil:
    cdef Uncertainty Robust
    cdef Uncertainty Optimistic
    cdef Uncertainty Average 
    
cdef extern from "../include/modeltools.hpp" namespace 'craam' nogil:
    void add_transition[Model](Model& mdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward)

from enum import Enum 

class UncertainSet(Enum):
    """
    Type of the solution to seek
    """
    Robust = 0
    Optimistic = 1
    Average = 2

DEFAULT_ITERS = 500

cdef class MDP:
    """
    Contains the definition of a standard MDP and related optimization algorithms.
    
    The states, actions, and outcomes are identified by consecutive ids, independently
    numbered for each type.
    
    Initialization requires the number of states.
    
    Parameters
    ----------
    statecount : int, optional (0)
        An estimate of the numeber of states (for pre-allocation). When more states
        are added, the estimate is readjusted.
    discount : double, optional (1.0)
        The discount factor
    """
    
    cdef shared_ptr[CMDP] thisptr
   
    """ Discount factor """
    cdef public double discount

    def __cinit__(self, int statecount = 0, double discount = 1.0):
        self.thisptr = make_shared[CMDP](statecount)

    def __init__(self, int statecount, double discount):
        self.discount = discount
        
    def __dealloc__(self):
        # this is probably not necessary
        self.thisptr.reset()
                
    cdef _check_value(self,valuefunction):
        if valuefunction.shape[0] > 0:
            if valuefunction.shape[0] != dereference(self.thisptr).state_count():
                raise ValueError('Value function dimensions must match the number of states.')

    cpdef copy(self):
        """ Makes a copy of the object """
        r = MDP(0, self.discount)
        r.thisptr.reset(new CMDP(dereference(self.thisptr)))
        return r

    cpdef add_transition(self, long fromid, long actionid, long toid, double probability, double reward):
        """
        Adds a single transition sample using outcome with id = 0. This function
        is meant to be used for constructing a non-robust MDP.

        Parameters
        ----------
        fromid : long
            Unique identifier of the source state of the transition 
        actionid : long
            Identifier of the action. It is unique for the given state
        toid : long
            Unique identifier of the target state of the transition
        probability : float
            Probability of the distribution
        reward : float
            Reward associated with the transition
        """        
        add_transition[CMDP](dereference(self.thisptr),fromid, actionid, 0, toid, probability, reward)

    cpdef long state_count(self):
        """ Current number of states """
        return dereference(self.thisptr).state_count()
        
    cpdef long action_count(self, long stateid):
        """
        Returns the number of actions
        
        Parameters
        ----------
        stateid : int
            Number of the state
        """
        return dereference(self.thisptr).get_state(stateid).action_count()
        
    cpdef long outcome_count(self, long stateid, long actionid):
        """
        Returns the number of outcomes
        
        Parameters
        ----------
        stateid : int
            Number of the state
        actionid : int
            Number of the action
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).outcome_count()

    cpdef long transition_count(self, long stateid, long actionid, long outcomeid):
        """
        Number of transitions (sparse transition probability) following a state,
        action, and outcome

        Parameters
        ----------
        stateid : int
            Number of the state
        actionid : int
            Number of the action
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).size()


    cpdef double get_reward(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ Returns the reward for the given state, action, and outcome """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_reward(sampleid)
        
    cpdef get_rewards(self, long stateid, long actionid, long outcomeid):
        """ Returns the reward for the given state, action, and outcome """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_rewards()

    cpdef long get_toid(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ Returns the target state for the given state, action, and outcome """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_indices()[sampleid]
        
    cpdef get_toids(self, long stateid, long actionid, long outcomeid):
        """ Returns the target state for the given state, action, and outcome """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_indices()

    cpdef double get_probability(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ Returns the probability for the given state, action, and outcome """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_probabilities()[sampleid]
    
    cpdef get_probabilities(self, long stateid, long actionid, long outcomeid):
        """ Returns the probability for the given state, action, and outcome """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_probabilities()

    cpdef set_reward(self, long stateid, long actionid, long outcomeid, long sampleid, double reward):
        """
        Sets the reward for the given state, action, outcome, and sample
        """
        dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).set_reward(sampleid, reward)
        
    cpdef long sample_count(self, long stateid, long actionid, long outcomeid):
        """
        Returns the number of samples (single-state transitions) for the action and outcome
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).size()
        
    cpdef vi_gs(self, int iterations=DEFAULT_ITERS, valuefunction = np.empty(0), \
                            double maxresidual=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        """
        
        self._check_value(valuefunction)
        cdef Uncertainty unc = Average
 
        cdef SolutionDscDsc sol = dereference(self.thisptr).vi_gs(unc,self.discount,\
                    valuefunction,iterations,maxresidual)

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations

    cpdef vi_jac(self, int iterations=DEFAULT_ITERS, valuefunction = np.empty(0), \
                                    double maxresidual=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        """

        self._check_value(valuefunction)
        cdef Uncertainty unc = Average

        cdef SolutionDscDsc sol = dereference(self.thisptr).vi_jac(unc,self.discount,\
                        valuefunction,iterations,maxresidual)

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations


    cpdef mpi_jac(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0), \
                                    double maxresidual = 0, long valiterations = 1000, int stype=0,
                                    double valresidual=-1):
        """
        Runs modified policy iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        valiterations : int, optional
            Maximal number of iterations for value function computation
        valresidual : double, optional 
            Maximal residual at which iterations of computing the value function 
            stop. Default is maxresidual / 2.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        """

        self._check_value(valuefunction)
        cdef Uncertainty unc = Average

        if valresidual < 0:
            valresidual = maxresidual / 2

        cdef SolutionDscDsc sol = dereference(self.thisptr).mpi_jac(unc,self.discount,\
                        valuefunction,iterations,maxresidual,valiterations,\
                        valresidual)

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations

    cpdef from_matrices(self, np.ndarray[double,ndim=3] transitions, np.ndarray[double,ndim=2] rewards, \
        double ignorethreshold = 1e-10):
        """
        Constructs an MDP from transition matrices, with uniform
        number of actions for each state. 
        
        The function replaces the current value of the object.
        
        Parameters
        ----------
        transitions : np.ndarray[double,double,double] (n x n x m)
            The last dimension represents the actions as defined by
            the parameter `action`. The first dimension represents
            the originating state in a transition and the second dimension represents
            the target state.
        rewards : np.ndarray[double, double] (n x m)
            The rewards for each state and action
        ignorethreshold : double, optional
            Any transition probability less than the threshold is ignored leading to 
            sparse representations. If not provided, no transitions are ignored
        """
        cdef int actioncount = transitions.shape[2]
        cdef int statecount = transitions.shape[0]

        # erase the current MDP object
        self.thisptr = make_shared[CMDP](statecount)

        if actioncount != rewards.shape[1]:
            raise ValueError('The number of actions must match 2nd dimension of rewards.')
        if statecount != transitions.shape[1] or statecount != rewards.shape[0]:
            raise ValueError('The number of states in transitions and rewards is inconsistent.')

        cdef int aoindex, fromid, toid
        cdef int actionid 
        cdef double transitionprob, rewardval

        for aoindex in range(actioncount):    
            for fromid in range(statecount):
                for toid in range(statecount):
                    actionid = aoindex
                    transitionprob = transitions[fromid,toid,aoindex]
                    if transitionprob <= ignorethreshold:
                        continue
                    rewardval = rewards[fromid,aoindex]
                    self.add_transition(fromid,actionid,toid,transitionprob,rewardval)

    cpdef to_matrices(self):
        """
        Build transitions matrices from the MDP.
        
        Number of states is ``n = |states|``. The number of available action-outcome
        pairs is ``m``.
        
        Returns
        ----------
        transitions : np.ndarray[double,double,double] (n x n x m)
            The last dimension represents the actions as defined by
            the parameter `action`. The first dimension represents
            the originating state in a transition and the second dimension represents
            the target state.
        rewards : np.ndarray[double, double] (n x m)
            The rewards for each state and action
        actions : np.ndarray[long] (m)
            The id of the action for the state
        """
        
        # compute the maximal number of actions

        cdef np.ndarray[double,ndim=3] transitions
        cdef np.ndarray[double,ndim=2] rewards

    cpdef to_json(self):
        """
        Returns a json representation of the MDP.  Use json.tool to pretty print.
        """
        return dereference(self.thisptr).to_json().decode('UTF-8')

cdef extern from "../include/Samples.hpp" namespace 'craam::msen':
    
    cdef cppclass CDiscreteSamples "craam::msen::DiscreteSamples":

        CDiscreteSamples();

        void add_initial(const long& decstate);
        void add_sample(const long& state_from, const long& action, const long& state_to, double reward, double weight, long step, long run);
        double mean_return(double discount);

        const vector[long]& get_states_from() const;
        const vector[long]& get_actions() const;
        const vector[long]& get_states_to() const;
        const vector[double]& get_rewards() const;
        const vector[double]& get_weights() const;
        const vector[long]& get_runs() const;
        const vector[long]& get_steps() const;
        const vector[long]& get_initial() const;


cdef class DiscreteSamples:
    """
    Collection of state to state transitions as well as samples of initial states. 
    All states and actions are identified by integers. 

    Sample weights are used to give proportional importance to samples when
    estimating transitions.

    Run references to the which execution of the simulator was used to get
    the particular sample and step references to the number of the step within the
    execution.
    """
    #TODO: When the functionality is added, just add the following doc
    # Class ``features.DiscreteSampleView`` can be used as a convenient method for assigning
    # state identifiers based on the equality between states.

    cdef CDiscreteSamples *_thisptr

    def __cinit__(self):
        self._thisptr = new CDiscreteSamples() 
        
    def __dealloc__(self):
        del self._thisptr        
        
    def __init__(self):
        """ 
        Creates empty sample dictionary and returns it.
        Can take arguments that describe the content of the samples.
        """
        pass
        
    def initialsamples(self):
        """
        Returns samples of initial decision states.
        This is separate from the transition samples.
        """
        return dereference(self._thisptr).get_initial();
        
    def get_states_from(self):
        """ Returns a list of all originating states (one for every sample)"""
        return dereference(self._thisptr).get_states_from()

    def get_actions(self):
        """ Returns a list of all actions (one for every sample)"""
        return dereference(self._thisptr).get_actions()

    def get_states_to(self):
        """ Returns a list of all destination states (one for every sample)"""
        return dereference(self._thisptr).get_states_to()

    def get_rewards(self):
        """ Returns a list of all rewards (one for every sample)"""
        return dereference(self._thisptr).get_rewards()

    def get_weights(self):
        """ Returns a list of all sample weights (one for every sample)"""
        return dereference(self._thisptr).get_weights()

    def get_runs(self):
        """ Returns a list of all run numbers (one for every sample)"""
        return dereference(self._thisptr).get_runs()

    def get_steps(self):
        """ Returns a list of all step numbers (one for every sample)"""
        return dereference(self._thisptr).get_steps()


cdef extern from "../include/Simulation.hpp" namespace 'craam::msen' nogil:

    CDiscreteSamples simulate[Model](Model& sim, ModelRandomPolicy pol, long horizon, long runs, long tran_limit, double prob_term, long seed);
    CDiscreteSamples simulate[Model](Model& sim, ModelRandomPolicy pol, long horizon, long runs);

    cdef cppclass ModelSimulator:
        ModelSimulator(const shared_ptr[CMDP] mdp, const CTransition& initial, long seed);
        ModelSimulator(const shared_ptr[CMDP] mdp, const CTransition& initial);

    cdef cppclass ModelRandomPolicy:
        
        ModelRandomPolicy(const ModelSimulator& sim, long seed);        
        ModelRandomPolicy(const ModelSimulator& sim);        

    cdef cppclass ModelDeterministicPolicy:
        
        ModelDeterministicPolicy(const ModelSimulator& sim, const indvec& actions);

cdef class Simulation:
    """
    Simulates an MDP.

    Constructs from and MDP object and an initial transition.

    Parameters
    ----------
    mdp : MDP
        Markov decision process source
    initial : np.ndarray
        Initial transition. The lentgth of the vector should correspond to the
        number of states.
    """
    cdef ModelSimulator *_thisptr

    def __cinit__(self, MDP mdp, np.ndarray[double] initial):

        if len(initial) != mdp.state_count():
            raise ValueError("Initial distribution must be as long as the number of MDP states, which is " + str(mdp.state_count()))

        cdef shared_ptr[CMDP] cmdp = mdp.thisptr

        self._thisptr = new ModelSimulator(cmdp, CTransition(initial)) 
                
    def __dealloc__(self):
        del self._thisptr        

    def simulate_random(self):
        """
        Simulates a uniformly random policy
        """

        # create a random policy
        cdef ModelRandomPolicy * rp =  new ModelRandomPolicy(dereference(self._thisptr))
        
        try:
            newsamples = DiscreteSamples()

            # TODO: make sure that this is moved through an rvalue
            newsamples._thisptr[0] = simulate(dereference(self._thisptr), dereference(rp), 10, 10);

            return newsamples

        finally:
            del rp


cdef extern from "../include/Simulation.hpp" namespace 'craam::msen' nogil:
    cdef cppclass CSampledMDP "craam::msen::SampledMDP":
        CSampledMDP();
        void add_samples(const CDiscreteSamples& samples);
        shared_ptr[CMDP] get_mdp_mod()


cdef class SampledMDP:
    """
    Constructs an MDP from provided samples
    """

    cdef shared_ptr[CSampledMDP] _thisptr
    
    def __cinit__(self):
        self._thisptr = make_shared[CSampledMDP]()

    cpdef add_samples(self, DiscreteSamples samples):
        """
        Adds samples to the MDP
        """
        dereference(self._thisptr).add_samples(dereference(samples._thisptr))

    cpdef get_mdp(self, discount):
        """
        Returns the MDP that was constructed from the samples.  If there 
        are more samples added, this MDP will be automatically modified
        """
        cdef MDP m = MDP(0, discount = discount)
        m.thisptr = dereference(self._thisptr).get_mdp_mod()
        return m

# distutils: language = c++
# distutils: libraries = craam
# distutils: library_dirs = ../lib 
# distutils: include_dirs = ../include 

cdef extern from "../include/RMDP.hpp" namespace 'craam' nogil:
    pair[vector[double],double] worstcase_l1(const vector[double] & z, \
                        const vector[double] & q, double t)

cpdef cworstcase_l1(np.ndarray[double] z, np.ndarray[double] q, double t):
    """
    o = cworstcase_l1(z,q,t)
    
    Computes the solution of:
    min_p   p^T * z
    s.t.    ||p - q|| <= t
            1^T p = 1
            p >= 0
            
    where o is the objective value
          
    Notes
    -----
    This implementation works in O(n log n) time because of the sort. Using
    quickselect to choose the right quantile would work in O(n) time.
    
    The parameter z may be a masked array. In that case, the distribution values 
    are normalized to the unmasked entries.
    """
    return worstcase_l1(z,q,t).second

cdef extern from "../include/RMDP.hpp" namespace 'craam' nogil:

    cdef cppclass SolutionDscProb:
        numvec valuefunction
        indvec policy
        vector[numvec] outcomes
        prec_t residual
        long iterations

    cdef cppclass RMDP_L1:
        RMDP_L1(long)
        RMDP_L1(const RMDP_L1&)
        RMDP_L1()

        size_t state_count() 

        void normalize()

        SolutionDscProb vi_jac(Uncertainty uncert, prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual) 


        SolutionDscProb vi_gs(Uncertainty uncert, prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual) 

        SolutionDscProb mpi_jac(Uncertainty uncert,
                    prec_t discount,
                    const numvec& valuefunction,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi)
        
        string to_json() const

cdef extern from "../include/modeltools.hpp" namespace 'craam' nogil:
    void set_outcome_thresholds[Model](Model& mdp, prec_t threshold)
    void set_uniform_outcome_dst[Model](Model& mdp)
    bool is_outcome_dst_normalized[Model](const Model& mdp)
    void normalize_outcome_dst[Model](Model& mdp)
    void set_outcome_dst[Model](Model& mdp, size_t stateid, size_t actionid, const numvec& dist)

cdef class RMDP:
    """
    Contains the definition of the robust MDP and related optimization algorithms.
    The algorithms can handle both robust and optimistic solutions.
    
    The states, actions, and outcomes are identified by consecutive ids, independently
    numbered for each type.
    
    Initialization requires the number of states.
    
    Parameters
    ----------
    statecount : int
        An estimate of the numeber of states (for pre-allocation). When more states
        are added, the estimate is readjusted.
    discount : double
        The discount factor
    """
    
    cdef shared_ptr[RMDP_L1] thisptr
    cdef public double discount

    def __cinit__(self, int statecount, double discount):
        self.thisptr = make_shared[RMDP_L1](statecount)

    def __init__(self, int statecount, double discount):
        self.discount = discount
        
    def __dealloc__(self):
        # this is probably not necessary
        self.thisptr.reset()
                
    cdef _check_value(self,valuefunction):
        if valuefunction.shape[0] > 0:
            if valuefunction.shape[0] != dereference(self.thisptr).state_count():
                raise ValueError('Value function dimensions must match the number of states.')

    cdef Uncertainty _convert_uncertainty(self,stype):
        cdef Uncertainty unc
        if stype == 0:
            unc = Robust
        elif stype == 1:
            unc = Optimistic
        elif stype == 2:
            unc = Average
        else:
            raise ValueError("Incorrect solution type '%s'." % stype )
        
        return unc

    cpdef add_transition(self, long fromid, long actionid, long outcomeid, long toid, double probability, double reward):
        """
        Adds a single transition sample using outcome with id = 0. This function
        is meant to be used for constructing a non-robust MDP.

        Parameters
        ----------
        fromid : long
            Unique identifier of the source state of the transition 
        actionid : long
            Identifier of the action. It is unique for the given state
        outcomeid : long
            Identifier of the outcome
        toid : long
            Unique identifier of the target state of the transition
        probability : float
            Probability of the distribution
        reward : float
            Reward associated with the transition
        """        
        add_transition[RMDP_L1](dereference(self.thisptr),fromid, actionid, outcomeid,
                                toid, probability, reward)

    cpdef set_distribution(self, long fromid, long actionid, np.ndarray[double] distribution):
        """
        Sets the base distribution over the states and the threshold
        
        Parameters
        ----------
        fromid : int
            Number of the originating state
        actionid : int
            Number of the actions
        distribution : np.ndarray
            Distributions over the outcomes (should be a correct distribution)
        """
        if abs(np.sum(distribution) - 1) > 0.001:
            raise ValueError('incorrect distribution (does not sum to one)', distribution)
        if np.min(distribution) < 0:
            raise ValueError('incorrect distribution (negative)', distribution)    

        set_outcome_dst[RMDP_L1](dereference(self.thisptr), fromid, actionid, distribution)
        
    cpdef set_uniform_distributions(self):
        """ Sets all the outcome distributions to be uniform. """
        set_uniform_outcome_dst[RMDP_L1](dereference(self.thisptr))

    cpdef set_uniform_thresholds(self, double threshold):
        """
        Sets the same threshold for all states.
        
        Can use ``self.set_distribution`` to set the thresholds individually for 
        each states and action.
        
        See Also
        --------
        self.set_distribution
        """
        set_outcome_thresholds[RMDP_L1](dereference(self.thisptr), \
                                        threshold)

    cpdef copy(self):
        """ Makes a copy of the object """
        r = RMDP(0, self.discount)
        r.thisptr.reset(new RMDP_L1(dereference(self.thisptr)))
        return r

    cpdef long state_count(self):
        """
        Returns the number of states
        """
        return dereference(self.thisptr).state_count()
        
        
    cpdef vi_gs(self, int iterations=DEFAULT_ITERS, valuefunction = np.empty(0), \
                            double maxresidual=0, int stype=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        stype : int  {0, 1, 2}, optional
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. UncertainSet.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcomes : np.ndarray
            Outcomes selected
        """
        
        self._check_value(valuefunction)
        cdef Uncertainty unc = self._convert_uncertainty(stype)
 
        cdef SolutionDscProb sol = dereference(self.thisptr).vi_gs(unc,self.discount,\
                    valuefunction,iterations,maxresidual)

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes


    cpdef vi_jac(self, int iterations=DEFAULT_ITERS,valuefunction = np.empty(0), \
                                    double maxresidual=0, int stype=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        stype : int  (0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average. One
            can use e.g. UncertainSet.Robust.value.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcomes : np.ndarray
            Outcomes selected
        """

        self._check_value(valuefunction)
        cdef Uncertainty unc = self._convert_uncertainty(stype)

        cdef SolutionDscProb sol = dereference(self.thisptr).vi_jac(unc,self.discount,\
                        valuefunction,iterations,maxresidual)

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes


    cpdef mpi_jac(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0), \
                                    double maxresidual = 0, long valiterations = 1000, int stype=0,
                                    double valresidual=-1):
        """
        Runs modified policy iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
        This is the parallel version of the update with values updates for all states
        simultaneously.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        maxresidual : double
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        valiterations : int, optional
            Maximal number of iterations for value function computation
        stype : int  (0, 1, 2}
            Robust (0) or optimistic (1) solution or (2) average solution. One
            can use e.g. UncertainSet.Robust.value.
        valresidual : double, optional 
            Maximal residual at which iterations of computing the value function 
            stop. Default is maxresidual / 2.
            
        Returns
        -------
        valuefunction : np.ndarray
            Optimized value function
        policy : np.ndarray
            Policy greedy for value function
        residual : double
            Residual for the value function
        iterations : int
            Number of iterations taken
        outcomes : np.ndarray
            Outcomes selected
        """

        self._check_value(valuefunction)
        cdef Uncertainty unc = self._convert_uncertainty(stype)

        if valresidual < 0:
            valresidual = maxresidual / 2

        cdef SolutionDscProb sol = dereference(self.thisptr).mpi_jac(unc,self.discount,\
                        valuefunction,iterations,maxresidual,valiterations,\
                        valresidual)

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes


    cpdef from_matrices(self, np.ndarray[double,ndim=3] transitions, np.ndarray[double,ndim=2] rewards, \
        np.ndarray[long] actions, np.ndarray[long] outcomes, double ignorethreshold = 1e-10):
        """
        Constructs an MDP from transition matrices. The function is meant to be
        called only once and cannot be used to re-initialize the transition 
        probabilities.
        
        Number of states is ``n = |states|``. The number of available action-outcome
        pairs is ``m``.
        
        Parameters
        ----------
        transitions : np.ndarray[double,double,double] (n x n x m)
            The last dimension represents the actions as defined by
            the parameter `action`. The first dimension represents
            the originating state in a transition and the second dimension represents
            the target state.
        rewards : np.ndarray[double, double] (n x m)
            The rewards for each state and action
        actions : np.ndarray[long] (m)
            The id of the action for the state
        outcomes : np.ndarray[long] (m)
            The id of the outcome for the state
        ignorethreshold : double, optional
            Any transition probability less than the threshold is ignored leading to 
            sparse representations. If not provided, no transitions are ignored
        """
        cdef int actioncount = len(actions) # really the number of action
        cdef int statecount = transitions.shape[0]

        if actioncount != transitions.shape[2] or actioncount != rewards.shape[1]:
            raise ValueError('The number of actions must match the 3rd dimension of transitions and the 2nd dimension of rewards.')
        if statecount != transitions.shape[1] or statecount != rewards.shape[0]:
            raise ValueError('The number of states in transitions and rewards is inconsistent.')
        if len(set(actions)) != actioncount:
            raise ValueError('The actions must be unique.')

        cdef int aoindex, fromid, toid
        cdef int actionid 
        cdef double transitionprob, rewardval

        for aoindex in range(actioncount):    
            for fromid in range(statecount):
                for toid in range(statecount):
                    actionid = actions[aoindex]
                    outcomeid = outcomes[aoindex]
                    transitionprob = transitions[fromid,toid,aoindex]
                    if transitionprob <= ignorethreshold:
                        continue
                    rewardval = rewards[fromid,aoindex]
                    self.add_transition(fromid, actionid, outcomeid, toid, transitionprob, rewardval)

    cpdef to_json(self):
        """
        Returns a json representation of the RMDP. Use json.tool to pretty print.
        """
        return dereference(self.thisptr).to_json().decode('UTF-8')



# ***************************************************************************
# *******    Implementable    *******
# ***************************************************************************

cdef extern from "../include/ImMDP.hpp" namespace 'craam::impl':
    
    cdef cppclass MDPI_R:
    
        MDPI_R(const CMDP& mdp, const indvec& observ2state, const CTransition& initial);

        vector[long] obspol2statepol(const vector[long]& obspol) except +;
        
        const RMDP_L1& get_robust_mdp() except +

        vector[long] solve_reweighted(long iterations, double discount) except +;
        vector[long] solve_robust(long iterations, double threshold, double discount) except +;
        
        double total_return(const vector[long]& obspol, double discount, double precision);
        
        void to_csv_file(const string& output_mdp, const string& output_state2obs, \
                        const string& output_initial, bool headers) except +;
    
        long state_count(); 
        long obs_count();

        unique_ptr[MDPI_R] from_csv_file(const string& input_mdp, \
                                            const string& input_state2obs, \
                                            const string& input_initial, \
                                            bool headers) except +;
                                            

cdef class MDPIR:
    """
    MDP with Implementability constraints. The implementability constraints
    require states within a single observation to have the same action
    chosen by the policy.

    Uses solution methods based on solving a robust MDP.

    Parameters
    ----------
    mdp : MDP
        Base MDP
    state2obs : np.ndarray
        Maps states to observation indexes. The observation index is 0-based
        and two states. The optimal 
    initial : np.ndarray
        The initial distribution
    copy_mdp : bool, optional (true)
        Whether to copy the MDP definition locally
    """

    cdef shared_ptr[MDPI_R] thisptr
    cdef double discount
    
    def __cinit__(self, MDP mdp, np.ndarray[long] state2obs, np.ndarray[double] initial, copy_mdp=True):

        cdef long states = mdp.state_count()
        
        if states != state2obs.size:
            raise ValueError('The number of MDP states must equal to the size of state2obs.')
        if state2obs.size != initial.size:
            raise ValueError('Sizes of state2obs and initial must be the same.')

        # construct the initial distribution
        cdef CTransition initial_t = CTransition(np.arange(states),initial,np.zeros(states))

        cdef indvec state2obs_c = state2obs
        if not copy_mdp:
            # this copies the MDP, it could be more efficient to just share the pointer
            # but then need to take care not to overwrite
            raise ValueError("Sharing MDP not yet supported")
        else:
            self.thisptr = make_shared[MDPI_R](dereference(mdp.thisptr), state2obs_c, initial_t)

    def __init__(self, MDP mdp, np.ndarray[long] state2obs, np.ndarray[double] initial):
        self.discount = mdp.discount

    def __dealloc__(self):
        pass

    def solve_reweighted(self, long iterations, double discount):
        """
        Solves the problem by reweighting the samples according to the current distribution
        
        Parameters
        ----------
        iterations : int
            Number of iterations
        discount : float
            Discount factor

        Returns
        -------
        out : list
            List of action indexes for observations
        """
        return dereference(self.thisptr).solve_reweighted(iterations, discount)

    def solve_robust(self, long iterations, double threshold, double discount):
        """
        Solves the problem by reweighting the samples according to the current distribution
        and computing a robust solution. The robustness is in terms of L1 norm and 
        determined by the threshold.
        
        Parameters
        ----------
        iterations : int
            Number of iterations
        threshold : double
            Bound on the L1 deviation probability
        discount : float
            Discount factor

        Returns
        -------
        out : list
            List of action indexes for observations
        """
        return dereference(self.thisptr).solve_robust(iterations, threshold, discount)


    def get_robust(self):
        """
        Returns the robust representation of the implementable MDP
        """
        cdef RMDP result = RMDP(0, self.discount)
        result.thisptr = make_shared[RMDP_L1](dereference(self.thisptr).get_robust_mdp())
        return result

    def obspol2statepol(self, np.ndarray[long] obspol):
        """
        Converts an observation policy to a state policy
        """
        return dereference(self.thisptr).obspol2statepol(obspol);

    def state_count(self):
        """ Number of states in the MDP """
        return dereference(self.thisptr).state_count()

    def obs_count(self):
        """ Number of observations in the interpretable MDP """
        return dereference(self.thisptr).obs_count()

    def total_return(self, np.ndarray[long] obspol):
        """ """
        assert len(obspol) == self.obs_count()
        return dereference(self.thisptr).total_return(obspol, self.discount, 1e-8)

    def to_csv(self, mdp_file, state2obs_file, initial_file, headers):
        """
        Saves the problem to a csv file
        """
        dereference(self.thisptr).to_csv_file(mdp_file, state2obs_file, initial_file, headers)

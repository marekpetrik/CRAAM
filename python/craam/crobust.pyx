# distutils: language = c++
# distutils: include_dirs = ../craam


"""
A suite of tools for sampling, solving and manipulating MDPs. Includes
robust and interpretable MDPs.

The main functionality is provided by the individual classes below:

- Solve MDPs: :py:class:`craam.MDP`
- Solve Robust MDPs: :py:class:`craam.RMDP`
- Simulate MDPs and generate samples: :py:class:`craam.SimulatorMDP`, :py:class:`craam.DiscreteSamples`
- Construct MDPs from samples: :py:class:`craam.SampledMDP`, :py:class:`craam.DiscreteSamples`
- Solve interpretable MDPs: :py:class:`craam.MDPIR`

This library is a thin Python wrapper around a C++ implementation.

References
----------

- Petrik, M., Subramanian, D. (2015). RAAM : The benefits of robustness in approximating aggregated MDPs in reinforcement learning. In Neural Information Processing Systems (NIPS).
- Petrik, M., & Luss, R. (2016). Interpretable Policies for Dynamic Product Recommendations. In Uncertainty in Artificial Intelligence (UAI).
"""

import numpy as np 
cimport numpy as np
from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp.utility cimport pair
from libcpp.memory cimport unique_ptr, shared_ptr
from libcpp cimport bool
import statistics
from collections import namedtuple 
from math import sqrt
import warnings 
from cython.operator import dereference


from libcpp.memory cimport make_shared
# Replace above for backwards compatibility with Cython 0.23 by text below
#cdef extern from "<memory>" namespace "std" nogil:
#    shared_ptr[T] make_shared[T](...) except +
#    unique_ptr[T] make_unique[T](...) # except +


cdef extern from "RMDP.hpp" namespace 'craam' nogil:
                                            
    ctypedef double prec_t
    ctypedef vector[double] numvec
    ctypedef vector[long] indvec
    ctypedef unsigned long size_t
                                            
    cdef cppclass Uncertainty:
        pass 


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
        CTransition& get_outcome()
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

        string to_json() const;

cdef extern from "algorithms/values.hpp" namespace "craam::algorithms" nogil:

    cdef cppclass Solution:
        numvec valuefunction
        indvec policy
        indvec outcomes
        prec_t residual
        long iterations

    Solution solve_vi(const CMDP& mdp, prec_t discount,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations,
                    prec_t maxresidual);

    Solution solve_mpi(const CMDP& mdp, prec_t discount,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi,
                    bool show_progress);
    

cdef extern from "modeltools.hpp" namespace 'craam' nogil:
    void add_transition[Model](Model& mdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward)

DEFAULT_ITERS = 500

import collections

SolutionTuple = namedtuple("Solution", ("valuefunction", "policy", "residual", "iterations")) 

cdef class MDP:
    """
    Contains the definition of a standard MDP and related optimization algorithms.
    
    The states, actions, and outcomes are identified by consecutive ids, independently
    numbered for each type.
    
    Initialization requires the number of states.
    
    Parameters
    ----------
    statecount : int, optional (0)
        An estimate of the number of states (for pre-allocation). When more states
        are added, the estimate is readjusted.
    discount : double, optional (1.0)
        The discount factor
    """
    
    cdef shared_ptr[CMDP] thisptr
   
    """ Discount factor """
    cdef public double discount

    def __cinit__(self, long statecount = 0, double discount = 1.0):
        self.thisptr = make_shared[CMDP](statecount)

    def __init__(self, long statecount, double discount):
        self.discount = discount

    def __dealloc__(self):
        # this is probably not necessary
        self.thisptr.reset()
                
    cdef _check_value(self,valuefunction):
        if valuefunction.shape[0] > 0:
            if valuefunction.shape[0] != dereference(self.thisptr).state_count():
                raise ValueError('Value function dimensions must match the number of states.')

    cdef _check_policy(self,policy):
        if policy.shape[0] > 0:
            if policy.shape[0] != dereference(self.thisptr).state_count():
                raise ValueError('Partial policy dimensions must match the number of states.')

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
        """ 
        Returns the number of states 
        """
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

    cpdef double get_reward(self, long stateid, long actionid, long sampleid):
        """ 
        Returns the reward for the given state, action, and outcome 

        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().get_reward(sampleid)
        
    cpdef get_rewards(self, long stateid, long actionid):
        """ 
        Returns the reward for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().get_rewards()

    cpdef long get_toid(self, long stateid, long actionid, long sampleid):
        """ 
        Returns the target state for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().get_indices()[sampleid]
        
        
    cpdef get_toids(self, long stateid, long actionid):
        """ 
        Returns the target state for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().get_indices()

    cpdef double get_probability(self, long stateid, long actionid, long sampleid):
        """ 
        Returns the probability for the given state, action, outcome, and index of a non-zero transition probability
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().get_probabilities()[sampleid]
    
    cpdef get_probabilities(self, long stateid, long actionid):
        """ 
        Returns the list of probabilities for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().get_probabilities()

    cpdef set_reward(self, long stateid, long actionid, long sampleid, double reward):
        """
        Sets the reward for the given state, action, outcome, and sample

        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        reward : double 
            New reward
        """
        dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().set_reward(sampleid, reward)
        
    cpdef long get_sample_count(self, long stateid, long actionid):
        """
        Returns the number of samples (single-state transitions) for the action and outcome

        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome().size()
        
    cpdef solve_vi(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                            policy = np.empty(0), double maxresidual=0):
        """
        Runs value iteration using the worst case (simplex) distribution for the 
        outcomes.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).

        Returns a namedtuple SolutionTuple.
        
        Parameters
        ----------
        iterations : int
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        policy : np.ndarray, optional
            Partial policy specification. Best action is chosen for states with 
            policy[state] = -1, otherwise the fixed value is used. 
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
        self._check_policy(policy)

        cdef Solution sol = solve_vi(dereference(self.thisptr), self.discount,\
                    valuefunction,policy,iterations,maxresidual)

        return SolutionTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations)


    cpdef mpi_jac(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                                    policy = np.empty(),
                                    double maxresidual = 0, long valiterations = -1, int stype=0,
                                    double valresidual=-1, bool show_progress = False):
        """
        Runs modified policy iteration with parallelized Jacobi valus updates
        
        Returns a namedtuple SolutionTuple.

        Parameters
        ----------
        iterations : int, optional
            Maximal number of iterations
        valuefunction : np.ndarray, optional
            The initial value function. Created automatically if not provided.            
        policy : np.ndarray, optional
            Partial policy specification. Best action is chosen for states with 
            policy[state] = -1, otherwise the fixed value is used. 
        maxresidual : double, optional
            Maximal residual at which the iterations stop. A negative value
            will ensure the necessary number of iterations.
        valiterations : int, optional
            Maximal number of iterations for value function computation. The same as iterations if omitted.
        valresidual : double, optional 
            Maximal residual at which iterations of computing the value function 
            stop. Default is maxresidual / 2.
        show_progress : bool
            Whether to report on the progress of the computation
            
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
        self._check_policy(policy)

        if valiterations <= 0: valiterations = iterations
        if valresidual < 0: valresidual = maxresidual / 2

        cdef Solution sol = solve_mpi(dereference(self.thisptr),self.discount,\
                        valuefunction,policy,iterations,maxresidual,valiterations,\
                        valresidual,show_progress)

        return SolutionTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations)

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

cdef extern from "../include/RMDP.hpp" namespace 'craam':
                                            
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

    cdef cppclass Transition:

        Transition() 
        Transition(const vector[long]& indices, const vector[double]& probabilities, const vector[double]& rewards);

        void set_reward(long sampleid, double reward) except +
        double get_reward(long sampleid) except +

        vector[double] probabilities_vector(unsigned long size) 

        vector[long]& get_indices() 
        vector[double]& get_probabilities()
        vector[double]& get_rewards() 
    
        size_t size() 

    cdef cppclass SolutionDscProb:
        pass

    cdef cppclass MDP:
        MDP(long)
        MDP()

        size_t state_count() 

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


    cdef cppclass RMDP_D:
        RMDP_D(long)
        RMDP_D()

        size_t state_count() 

        SolutionDscDsc vi_jac(Uncertainty uncert, prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual) 


        SolutionDscDsc vi_gs(Uncertainty uncert, prec_t discount,
                        const numvec& valuefunction,
                        unsigned long iterations,
                        prec_t maxresidual) 

        SolutionDscDsc mpi_jac(Uncertainty uncert,
                    prec_t discount,
                    const numvec& valuefunction,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi)

cdef extern from "../include/RMDP.hpp" namespace 'craam::Uncertainty':
    cdef Uncertainty Robust
    cdef Uncertainty Optimistic
    cdef Uncertainty Average 
    
cdef extern from "../include/modeltools.hpp" namespace 'craam':

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

cdef class SMDP:
    """
    Contains the definition of a standard MDP and related optimization algorithms.
    
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
    
    cdef shared_ptr[MDP] thisptr
    cdef public double discount

    def __cinit__(self, int statecount, double discount):
        self.thisptr = make_shared[MDP](statecount)

    def __init__(self, int statecount, double discount):
        self.discount = discount
        
    def __dealloc__(self):
        pass
        #del self.thisptr
                
    cdef _check_value(self,valuefunction):
        if valuefunction.shape[0] > 0:
            if valuefunction.shape[0] != dereference(self.thisptr).state_count():
                raise ValueError('Value function dimensions must match the number of states.')

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
        add_transition[MDP](dereference(self.thisptr),fromid, actionid, 0, toid, probability, reward)

        
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
        np.ndarray[long] actions, double ignorethreshold = 1e-10):
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
                    transitionprob = transitions[fromid,toid,aoindex]
                    if transitionprob <= ignorethreshold:
                        continue
                    rewardval = rewards[fromid,aoindex]
                    self.add_transition(fromid,actionid,toid,transitionprob,rewardval)

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
    
    cdef RMDP_D *thisptr
    cdef public double discount

    def __cinit__(self, int statecount, double discount):
        self.thisptr = new RMDP_D(statecount)

    def __init__(self, int statecount, double discount):
        self.discount = discount
        
    def __dealloc__(self):
        del self.thisptr
                
    cdef _check_value(self,valuefunction):
        if valuefunction.shape[0] > 0:
            if valuefunction.shape[0] != self.thisptr.state_count():
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
        add_transition[RMDP_D](self.thisptr[0],fromid, actionid, outcomeid, toid, probability, reward)

        
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
        stype : int  {0, 1, 2}
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
 
        cdef SolutionDscDsc sol = self.thisptr.vi_gs(unc,self.discount,\
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

        cdef SolutionDscDsc sol = self.thisptr.vi_jac(unc,self.discount,\
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

        cdef SolutionDscDsc sol = self.thisptr.mpi_jac(unc,self.discount,\
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


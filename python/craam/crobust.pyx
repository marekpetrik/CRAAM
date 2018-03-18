# distutils: language = c++


"""
A suite of tools for sampling, solving and manipulating MDPs. Includes
robust and interpretable MDPs.

The main functionality is provided by the individual classes below:

- Model and solve MDPs: :py:class:`craam.MDP`
- Model and solve RMDPs (with outcomes): :py:class:`craam.RMDP`
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


cdef extern from "craam/RMDP.hpp" namespace 'craam' nogil:
                                            
    ctypedef double prec_t
    ctypedef vector[double] numvec
    ctypedef vector[long] indvec
    ctypedef unsigned long size_t
                                            
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

cdef extern from "craam/algorithms/values.hpp" namespace "craam::algorithms" nogil:

    cdef cppclass Solution:
        numvec valuefunction
        indvec policy
        indvec outcomes
        prec_t residual
        long iterations

    Solution csolve_vi_mdp "craam::algorithms::solve_vi"(const CMDP& mdp, prec_t discount,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations,
                    prec_t maxresidual) except +;

    Solution csolve_mpi_mdp "craam::algorithms::solve_mpi"(const CMDP& mdp, prec_t discount,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi,
                    bool show_progress) except +;
    
SolutionRobustTuple = namedtuple("Solution", ("valuefunction", "policy", "residual", "iterations","natpolicy")) 

cdef extern from "craam/algorithms/robust_values.hpp" namespace 'craam::algorithms' nogil:

    cdef cppclass SolutionRobust:
        numvec valuefunction
        indvec policy
        vector[numvec] natpolicy
        prec_t residual
        long iterations

    ctypedef pair[numvec, prec_t] vec_scal_t 
    ctypedef vec_scal_t (*NatureResponse)(const numvec& v, const numvec& p, prec_t threshold)

    cdef NatureResponse string_to_nature(string s);

cdef extern from "craam/algorithms/robust_values.hpp" namespace 'craam::algorithms' nogil:

    vector[vector[double]] pack_thresholds(indvec states, indvec actions, numvec values) except +


    SolutionRobust crsolve_vi_mdp "craam::algorithms::rsolve_vi"(CMDP& mdp, prec_t discount,
                    NatureResponse nature, const vector[vector[double]]& thresholds,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations,
                    prec_t maxresidual) except +


    SolutionRobust crsolve_mpi_mdp "craam::algorithms::rsolve_mpi"(
                    CMDP& mdp, prec_t discount,
                    NatureResponse nature, const vector[vector[double]]& thresholds,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi,
                    bool show_progress) except +

cdef extern from "craam/modeltools.hpp" namespace 'craam' nogil:
    void add_transition[Model](Model& mdp, long fromid, long actionid, long outcomeid, long toid, prec_t probability, prec_t reward)

DEFAULT_ITERS = 500

import collections

SolutionTuple = namedtuple("Solution", ("valuefunction", "policy", "residual", "iterations")) 

cdef class MDP:
    """
    Contains the definition of a standard MDP  model and related optimization algorithms. It supports 
    both regular and robust solutions.
    
    The states, actions, and outcomes are identified by consecutive ids, independently
    numbered for each type.
    
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
        cdef long actioncount = transitions.shape[2]
        cdef long statecount = transitions.shape[0]

        # erase the current MDP object
        self.thisptr = make_shared[CMDP](statecount)

        if actioncount != rewards.shape[1]:
            raise ValueError('The number of actions must match 2nd dimension of rewards.')
        if statecount != transitions.shape[1] or statecount != rewards.shape[0]:
            raise ValueError('The number of states in transitions and rewards is inconsistent.')

        cdef long aoindex, fromid, toid
        cdef long actionid 
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
        
        Must have the same number of action for each state. Output is also given
        for invalid actions.

        Returns
        ----------
        transitions : np.ndarray[double,double,double] (n x n x m)
            The last dimension represents the actions as defined by
            the parameter `action`. The first dimension represents
            the originating state in a transition and the second dimension represents
            the target state.
        rewards : np.ndarray[double, double] (n x m)
            The rewards for each state and action
        """

        cdef long state_count = dereference(self.thisptr).state_count() 

        if state_count == 0:
            return None,None

        cdef size_t action_count = dereference(self.thisptr).get_state(0).action_count()

        cdef long s, s1i, s2i, ai 

        for si in range(state_count):
            if dereference(self.thisptr).get_state(si).action_count() != action_count:
                raise ValueError("Not the same number of actions for each state: " + str(si))

        cdef np.ndarray[double,ndim=3] transitions = np.zeros((state_count, state_count, action_count))
        cdef np.ndarray[double,ndim=2] rewards = np.zeros((state_count, action_count))

        cdef long sample_count, sci

        cdef double prob, rew

        for s1i in range(state_count):
            for ai in range(action_count):
                sample_count = self.get_sample_count(s1i,ai)
                for sci in range(sample_count):
                    s2i = self.get_toid(s1i,ai,sci)
                    prob = self.get_probability(s1i,ai,sci)
                    rew = self.get_reward(s1i,ai,sci)

                    transitions[s1i, s2i, ai] = prob
                    rewards[s1i, ai] += prob * rew

        return transitions, rewards

    cpdef to_json(self):
        """
        Returns a json representation of the MDP.  Use json.tool to pretty print.
        """
        return dereference(self.thisptr).to_json().decode('UTF-8')

    cpdef solve_vi(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                            policy = np.empty(0), double maxresidual=0):
        """
        Runs regular value iteration (no robustness or nature response).
        
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

        cdef Solution sol = csolve_vi_mdp(dereference(self.thisptr), self.discount,\
                    valuefunction,policy,iterations,maxresidual)

        return SolutionTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations)


    cpdef solve_mpi(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                                    policy = np.empty(0),
                                    double maxresidual = 0, long valiterations = -1, 
                                    double valresidual=-1, bool show_progress = False):
        """
        Runs regular modified policy iteration with parallelized Jacobi valus updates. No robustness
        or nature response.
        
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

        cdef Solution sol = csolve_mpi_mdp(dereference(self.thisptr),self.discount,\
                        valuefunction,policy,iterations,maxresidual,valiterations,\
                        valresidual,show_progress)

        return SolutionTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations)

    cpdef rsolve_vi(self, nature, thresholds, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                            policy = np.empty(0), double maxresidual=0):
        """
        Runs robust value iteration.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).

        Returns a namedtuple SolutionRobustTuple.
        
        Parameters
        ----------
        nature : string
            Type of response of nature. See choose_nature for supported values.
        thresholds : (stateids, actionids, thresholds values)
            Each entry represents the threshold for each state and action. The threshold
            should be provided for each state and action value; the ones that are not 
            specified are undefined.
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
        natpolicy : np.ndarray
            Best responses selected by nature
        """
        
        self._check_value(valuefunction)
        self._check_policy(policy)

        cdef SolutionRobust sol = crsolve_vi_mdp(dereference(self.thisptr), self.discount,\
                        string_to_nature(nature), pack_thresholds(thresholds[0], thresholds[1], thresholds[2]),
                        valuefunction,policy,iterations,maxresidual)

        return SolutionRobustTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations,sol.natpolicy)


    cpdef rsolve_mpi(self, nature, thresholds, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                                    policy = np.empty(0),
                                    double maxresidual = 0, long valiterations = -1,  
                                    double valresidual=-1, bool show_progress = False):
        """
        Runs robust modified policy iteration with parallelized Jacobi value updates.         

        Returns a namedtuple SolutionRobustTuple.

        Parameters
        ----------
        nature : string
            Type of response of nature. See choose_nature for supported values.
        thresholds : (stateids, actionids, thresholds values)
            Each entry represents the threshold for each state and action. The threshold
            should be provided for each state and action value; the ones that are not 
            specified are undefined.
        natpolicy : np.ndarray
            Best responses selected by nature
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
        natpolicy : np.ndarray
            Best responses selected by nature
        """

        self._check_value(valuefunction)
        self._check_policy(policy)

        if valiterations <= 0: valiterations = iterations
        if valresidual < 0: valresidual = maxresidual / 2

        cdef SolutionRobust sol = crsolve_mpi_mdp(dereference(self.thisptr),self.discount,\
                        string_to_nature(nature), pack_thresholds(thresholds[0], thresholds[1], thresholds[2]),
                        valuefunction,policy,iterations,maxresidual,valiterations,\
                        valresidual,show_progress)

        return SolutionRobustTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.natpolicy)
cdef extern from "craam/Samples.hpp" namespace 'craam::msen':
    
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
        
    def add_sample(self, long state_from, long action, long state_to, long reward, double weight=1.0, long step=0, long run=-1):
        """
        Adds a new individual sample to the collection

        Parameters
        ----------
        state_from : int
            Originating state
        action: int
            Action taken
        state_to : int
            Destination step
        reward : double
            Reward received
        weight : double, optional
            Relative weight of the sample
        step : int, optional
            Index of the sample within a single sequence (0-based)
        run : int, optional
            Numerical identifier of the current run (sequence)
        """
        dereference(self._thisptr).add_sample(state_from, action, state_to, reward, weight, step, run)

    def add_initial(self, long stateid):
        """
        Adds the state as a sample from the initial distribution
        """
        dereference(self._thisptr).add_initial(stateid)

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


cdef extern from "craam/Simulation.hpp" namespace 'craam::msen' nogil:

    cdef cppclass ModelSimulator:
        ModelSimulator(const shared_ptr[CMDP] mdp, const CTransition& initial, long seed);
        ModelSimulator(const shared_ptr[CMDP] mdp, const CTransition& initial);

    # this is a fake class just to fool cython to make the right calls
    cdef cppclass Policy:
        pass

    cdef cppclass ModelRandomPolicy(Policy):
        ModelRandomPolicy(const ModelSimulator& sim, long seed);        
        ModelRandomPolicy(const ModelSimulator& sim);        

    cdef cppclass ModelDeterministicPolicy(Policy):
        ModelDeterministicPolicy(const ModelSimulator& sim, const indvec& actions);

    CDiscreteSamples simulate[Model](Model& sim, Policy pol, long horizon, long runs, long tran_limit, double prob_term, long seed);
    CDiscreteSamples simulate[Model](Model& sim, Policy pol, long horizon, long runs, long tran_limit, double prob_term);

    pair[indvec, numvec] simulate_return[Model](Model& sim, double discount, Policy pol, long horizon, long runs, double prob_term, long seed);
    pair[indvec, numvec] simulate_return[Model](Model& sim, double discount, Policy pol, long horizon, long runs, double prob_term);

cdef class SimulatorMDP:
    """
    Simulates state evolution of an MDP for a given policy.

    Parameters
    ----------
    mdp : MDP
        Markov decision process that governs the simulation.
    initial : np.ndarray
        Probability distribution for the initial state. 
        Its length must match the number of states and must be 
        a valid distribution.
    """
    cdef ModelSimulator *_thisptr
    cdef long _state_count
    cdef double _discount

    def __cinit__(self, MDP mdp, np.ndarray[double] initial):

        if len(initial) != mdp.state_count():
            raise ValueError("Initial distribution must be as long as the number of MDP states, which is " + str(mdp.state_count()))

        cdef shared_ptr[CMDP] cmdp = mdp.thisptr
        # cache the number of state to check that the provided policy is correct
        self._state_count = dereference(cmdp).state_count()
        self._thisptr = new ModelSimulator(cmdp, CTransition(initial)) 
        self._discount = mdp.discount
                
    def __dealloc__(self):
        del self._thisptr        
    
    def state_count(self):
        "Number of states in the underlying MDP."""
        return self._state_count

    def simulate_random(self, horizon, runs, tran_limit=0, prob_term=0.0):
        """
        Simulates a uniformly random policy
    
        Parameters
        ----------
        horizon : int 
            Simulation horizon
        runs : int
            Number of simulation runs
        tran_limit : int, optional 
            Limit on the total number of transitions generated
            across all the runs. The simulation stops once 
            this number is reached.
        prob_term : double, optional
            Probability of terminating after each transitions. Used
            to simulate the discount factor.

        Returns
        -------
        out : DiscreteSamples
        """
        cdef ModelRandomPolicy * rp = \
                new ModelRandomPolicy(dereference(self._thisptr))
        
        try:
            newsamples = DiscreteSamples()
            newsamples._thisptr[0] = simulate[ModelSimulator](dereference(self._thisptr), dereference(rp), horizon, runs, tran_limit, prob_term);
            return newsamples
        finally:
            del rp

    def simulate_policy(self, np.ndarray[long] policy, horizon, runs, tran_limit=0, prob_term=0.0):
        """
        Simulates a policy

        Parameters
        ----------
        policy : np.ndarray[long]
            Policy used for the simulation. Must be as long as
            the number of states. Each entry marks the index
            of the action to take (0-based)
        horizon : int 
            Simulation horizon
        runs : int
            Number of simulation runs
        tran_limit : int, optional 
            Limit on the total number of transitions generated
            across all the runs. The simulation stops once 
            this number is reached.
        prob_term : double, optional
            Probability of terminating after each transitions. Used
            to simulate the discount factor.

        Returns
        -------
        out : DiscreteSamples
        """

        if policy.shape[0] != self._state_count:
            raise ValueError("Policy size must match the number of states " + str(self._state_count))

        cdef ModelDeterministicPolicy * rp = \
                new ModelDeterministicPolicy(dereference(self._thisptr), policy)
        
        try:
            newsamples = DiscreteSamples()
            newsamples._thisptr[0] = simulate[ModelSimulator](dereference(self._thisptr), dereference(rp), horizon, runs, tran_limit, prob_term);
            return newsamples
        finally:
            del rp
        
    def simulate_policy_return(self, np.ndarray[long] policy, horizon, runs, discount=None, prob_term=0.0):
        """
        Simulates a policy

        Parameters
        ----------
        policy : np.ndarray[long]
            Policy used for the simulation. Must be as long as
            the number of states. Each entry marks the index
            of the action to take (0-based)
        horizon : int 
            Simulation horizon
        runs : int
            Number of simulation runs
        discount : double, optional
            Discount factor, uses the one from the MDP is not provided
        prob_term : double, optional
            Probability of terminating after each transitions. Used
            to simulate the discount factor.

        Returns
        -------
        states : np.ndarray[long]
            State for which returns are available
        returns : np.ndarray[long]
            Returns for those states
        """

        if policy.shape[0] != self._state_count:
            raise ValueError("Policy size must match the number of states " + str(self._state_count))

        if discount is None:
            discount = self._discount

        cdef pair[indvec,numvec] result
        cdef ModelDeterministicPolicy * rp = \
                new ModelDeterministicPolicy(dereference(self._thisptr), policy)
        try:
            result = simulate_return[ModelSimulator](dereference(self._thisptr), \
                        discount, dereference(rp), horizon, runs, prob_term);
            
            return result.first, result.second
        finally:
            del rp

cdef extern from "craam/simulators/inventory_simulation.hpp" namespace 'craam::msen' nogil:

    cdef cppclass CInventorySimulator "craam::msen::InventorySimulator":
        CInventorySimulator(long initial, double prior_mean, double prior_std, double demand_std, double purchase_cost,
                       double sale_price, double delivery_cost, double holding_cost, double backlog_cost,
                       long max_inventory, long max_backlog, long max_order, long seed);
        CInventorySimulator(long initial, double prior_mean, double prior_std, double demand_std, double purchase_cost,
                       double sale_price, long max_inventory, long seed);
        long init_state();
        void init_demand_distribution();

    # this is a fake class just to fool cython to make the right calls
    cdef cppclass Policy:
        pass

    cdef cppclass ModelInventoryPolicy(Policy):
        ModelInventoryPolicy(const CInventorySimulator& sim, long max_inventory, long seed);               

cdef class SimulatorInventory:
    """
    Simulates state evolution of an inventory

    Example usage:
    from craam import crobust
    initial, max_inventory, purchase_cost, sale_price, prior_mean, prior_std, \
                        demand_std, rand_seed = 0, 50, 2.0, 3.0, 10.0, 4.0, 6.0, 3
    horizon, runs = 10, 5
    inventory_simulator = crobust.SimulatorInventory(initial, prior_mean, prior_std, demand_std, purchase_cost, \
                                                        sale_price, max_inventory, rand_seed)
    samples = inventory_simulator.simulate_inventory(horizon, runs)
    
    
    Parameters
    ----------
    initial : long
        Initial inventory level
    prior_mean : double
        prior mean value for the demand distriubtion
    prior_std : double
        prior standard deviation for the demand distribution
    demand_std : double
        Known true standard deviation of the demand
    purcahse_cost : double
        Cost of purchasing each unit of product
    sale_price : double
        Selling price of each unit of item
    max_inventory : long
        Maximum possible level of the inventory
    seed : long
        Seed for the random number generator
    """
    cdef CInventorySimulator *_thisptr
    cdef long max_inventory
    cdef long seed
    
    def __cinit__(self, long initial, double prior_mean, double prior_std, double demand_std, double purchase_cost,
                       double sale_price, long max_inventory, long seed):
        self.seed = seed
        self.max_inventory = max_inventory
        self._thisptr = new CInventorySimulator(initial, prior_mean, prior_std, demand_std, purchase_cost, sale_price, max_inventory, seed)
    
    def __dealloc__(self):
        del self._thisptr

    def simulate_inventory(self, horizon, runs, tran_limit=0, prob_term=0.0):
        """
        Simulates a s-S policy based on the current inventory level & required inventory level
    
        Parameters
        ----------
        horizon : int 
            Simulation horizon
        runs : int
            Number of simulation runs
        tran_limit : int, optional 
            Limit on the total number of transitions generated
            across all the runs. The simulation stops once 
            this number is reached.
        prob_term : double, optional
            Probability of terminating after each transitions. Used
            to simulate the discount factor.

        Returns
        -------
        out : DiscreteSamples
        """
        cdef ModelInventoryPolicy * rp = \
                new ModelInventoryPolicy(dereference(self._thisptr), self.max_inventory, self.seed)
        
        try:
            newsamples = DiscreteSamples()
            newsamples._thisptr[0] = simulate[CInventorySimulator](dereference(self._thisptr), dereference(rp), horizon, runs, tran_limit, prob_term, self.seed);
            return newsamples
        finally:
            del rp

cdef extern from "craam/simulators/invasive_species_simulation.hpp" namespace 'craam::msen' nogil:

    cdef cppclass CInvasiveSpeciesSimulator "craam::msen::InvasiveSpeciesSimulator":
        CInvasiveSpeciesSimulator(long initial_population, long carrying_capacity, double mean_growth_rate, double std_growth_rate,
                       double std_observation, double beta_1, double beta_2, long n_hat, long seed);
        long init_state();

    # this is a fake class just to fool cython to make the right calls
    cdef cppclass Policy:
        pass

    cdef cppclass ModelInvasiveSpeciesPolicy(Policy):
        ModelInvasiveSpeciesPolicy(const CInvasiveSpeciesSimulator& sim, double threshold_control, double prob_control, long seed);               

cdef class SimulatorSpecies:
    """
    Simulates population for invasive species.
    
    Example usage:
    from craam import crobust
    initial_population, carrying_capacity, mean_growth_rate, std_growth_rate, std_observation, \
    beta_1, beta_2, n_hat, threshold_control, prob_control, seed = 30, 1000, 1.02, 0.02, 10, 0.001, -0.0000021, 300, 0, 0.5, 3
    
    species_simulator = crobust.SimulatorSpecies(initial_population, carrying_capacity, mean_growth_rate, std_growth_rate, \
                                                    std_observation, beta_1, beta_2, n_hat, threshold_control, prob_control, seed)
    samples = species_simulator.simulate_species(horizon, runs)


    Parameters
    ----------
    initial_population : long
        Initial population level
    carrying_capacity : long
        Maximum carrying capacity of the ecosystem
    mean_growth_rate : double
        Mean growth rate of the population
    std_growth_rate : double
        Standard deviation of the population growth
    std_observation : double
        Standard deviation of the observation from actual population
    beta_1 : double
        Linear impact of the control
    beta_2 : double
        quadratic impact of the control
    n_hat : long
        point at which effectiveness of the control peaks
    threshold_control : double
        threshold at which the control is applied
    prob_control : double
        probability that the control is applied
    seed : long
        Seed for the random number generator
    """
    cdef CInvasiveSpeciesSimulator *_thisptr
    cdef long seed
    cdef double threshold_control
    cdef double prob_control
    
    def __cinit__(self, long initial_population, long carrying_capacity, double mean_growth_rate, double std_growth_rate,
                       double std_observation, double beta_1, double beta_2, long n_hat, double threshold_control,
                       double prob_control, long seed):
        self.seed = seed
        self.threshold_control = threshold_control
        self.prob_control = prob_control
        self._thisptr = new CInvasiveSpeciesSimulator(initial_population, carrying_capacity, mean_growth_rate, std_growth_rate, std_observation, beta_1, beta_2, n_hat, seed)
    
    def __dealloc__(self):
        del self._thisptr

    def simulate_species(self, horizon, runs, tran_limit=0, prob_term=0.0):
        """
        Simulates a s-S policy based on the current inventory level & required inventory level
    
        Parameters
        ----------
        horizon : int 
            Simulation horizon
        runs : int
            Number of simulation runs
        tran_limit : int, optional 
            Limit on the total number of transitions generated
            across all the runs. The simulation stops once 
            this number is reached.
        prob_term : double, optional
            Probability of terminating after each transitions. Used
            to simulate the discount factor.

        Returns
        -------
        out : DiscreteSamples
        """
        cdef ModelInvasiveSpeciesPolicy * rp = \
                new ModelInvasiveSpeciesPolicy(dereference(self._thisptr), self.threshold_control, self.prob_control, self.seed)
        
        try:
            newsamples = DiscreteSamples()
            newsamples._thisptr[0] = simulate[CInvasiveSpeciesSimulator](dereference(self._thisptr), dereference(rp), horizon, runs, tran_limit, prob_term, self.seed);
            return newsamples
        finally:
            del rp

cdef extern from "craam/Simulation.hpp" namespace 'craam::msen' nogil:
    cdef cppclass CSampledMDP "craam::msen::SampledMDP":
        CSampledMDP();
        void add_samples(const CDiscreteSamples& samples);
        shared_ptr[CMDP] get_mdp_mod()
        vector[vector[double]] get_state_action_weights()
        CTransition get_initial()
        long state_count();


cdef class SampledMDP:
    """
    Constructs an MDP from samples: :py:class:`DiscreteSamples`.

    Samples can be added multiple times and the MDP is updated 
    automatically.
    """

    cdef shared_ptr[CSampledMDP] _thisptr
    
    def __cinit__(self):
        self._thisptr = make_shared[CSampledMDP]()

    cpdef add_samples(self, DiscreteSamples samples):
        """
        Adds samples to the MDP.

        Parameters
        ----------
        samples : craam.DiscreteSamples
            Source of samples to be added to transition probabilties
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
        
    cpdef get_state_action_weights(self):
        return dereference(self._thisptr).get_state_action_weights()

    cpdef get_initial(self):
        """
        Returns the initial distribution inferred from samples
        """
        cdef long int state_count = dereference(self._thisptr).state_count()
        cdef CTransition t = dereference(self._thisptr).get_initial()
        return np.array(t.probabilities_vector(state_count))
            
  
cdef extern from "craam/fastopt.hpp" namespace 'craam' nogil:
    pair[numvec,double] c_worstcase_l1 "craam::worstcase_l1" (const vector[double] & z, \
                        const vector[double] & q, double t)


cpdef worstcase_l1(np.ndarray[double] z, np.ndarray[double] q, double t):
    """
    Computes a worstcase distribution subject to an L1 constraint

    o = worstcase_l1(z,q,t)
    
    Computes the solution of:
    min_p   p^T * z
    s.t.    ||p - q|| <= t
            1^T p = 1
            p >= 0
            
    where o is the objective value

    See Also
    --------
    worstcase_l1_dst returns also the optimal solution not only the optimal value
          
    Notes
    -----
    This implementation works in O(n log n) time because of the sort. Using
    quickselect to choose the correct quantile would work in O(n) time.
    """
    return c_worstcase_l1(z,q,t).second

cpdef cworstcase_l1(np.ndarray[double] z, np.ndarray[double] q, double t):
    """
    DEPRECATED: use worstcase_l1 instead
    """
    return worstcase_l1(z,q,t)

def worstcase_l1_dst(np.ndarray[double] z, np.ndarray[double] q, double t):
    """
    Computes a worstcase distribution subject to an L1 constraint

    p,o = worstcase_l1_dst(z,q,t)
    
    Computes the solution of:
    min_p   p^T * z
    s.t.    ||p - q|| <= t
            1^T p = 1
            p >= 0
            
    where o is the objective value
          
    Returns
    -------
    p_opt : np.ndarray
        Optimal solution
    f_opt : float
        Optimal objective value
          
    Notes
    -----
    This implementation works in O(n log n) time because of the sort. Using
    quickselect to choose the correct quantile would work in O(n) time.
    """
    cdef pair[numvec, double] x = c_worstcase_l1(z,q,t)
    return x.first, x.second





cdef extern from "craam/algorithms/values.hpp" namespace 'craam::algorithms' nogil:

    Solution csolve_vi_rmdp "craam::algorithms::solve_vi"(CRMDP& mdp, prec_t discount,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations,
                    prec_t maxresidual) except +;

    Solution csolve_mpi_rmdp "craam::algorithms::solve_mpi"(CRMDP& mdp, prec_t discount,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi,
                    bool show_progress) except +;

cdef extern from "craam/algorithms/robust_values.hpp" namespace 'craam::algorithms' nogil:
    SolutionRobust crsolve_vi "craam::algorithms::rsolve_vi"(CRMDP& mdp, prec_t discount,
                    NatureResponse nature, const vector[vector[double]]& thresholds,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations,
                    prec_t maxresidual) except +


    SolutionRobust crsolve_mpi "craam::algorithms::rsolve_mpi"(
                    CRMDP& mdp, prec_t discount,
                    NatureResponse nature, const vector[vector[double]]& thresholds,
                    const numvec& valuefunction,
                    const indvec& policy,
                    unsigned long iterations_pi,
                    prec_t maxresidual_pi,
                    unsigned long iterations_vi,
                    prec_t maxresidual_vi,
                    bool show_progress) except +

def choose_nature(nature_name):
    """
    The function does not do anything; it only serves as a documentation point.

    These nature values are currently supported:

    - robust_unbounded
    - optimistic_unbounded
    - robust_l1
    - optimistic_l1
    """
    pass


cdef extern from "craam/RMDP.hpp" namespace 'craam' nogil:

    cdef cppclass CL1OutcomeAction "craam::WeightedOutcomeAction":
        CTransition& get_outcome(long outcomeid)
        size_t outcome_count()


    cdef cppclass CL1RobustState "craam::RobustState":
        CL1OutcomeAction& get_action(long actionid)
        size_t action_count()

    cdef cppclass CRMDP "craam::RMDP":
        CRMDP(long)
        CRMDP(const CRMDP&)
        CRMDP()

        size_t state_count() 
        CL1RobustState& get_state(long stateid)

        void normalize()
                
        string to_json() const

cdef extern from "craam/modeltools.hpp" namespace 'craam' nogil:
    void set_uniform_outcome_dst[Model](Model& mdp)
    bool is_outcome_dst_normalized[Model](const Model& mdp)
    void normalize_outcome_dst[Model](Model& mdp)
    void set_outcome_dst[Model](Model& mdp, size_t stateid, size_t actionid, const numvec& dist)
    CRMDP robustify(const CMDP& mdp, bool allowzeros)


cdef class RMDP:
    """
    An extended MDP class that supports not only states and actions, but also adds outcomes which allow
    for a more convenient formulation of low-rank ambiguities and correlations between states and actions.

    The robust solution to this class will satisfy the following Bellman optimality equation:

    \f[v(s) = \max_{a \in \mathcal{A}} \min_{o \in \mathcal{O}} \sum_{s\in\mathcal{S}} ( r(s,a,o,s') + \gamma P(s,a,o,s') v(s') ) ~.\f]

    Here, \f$\mathcal{S}\f$ are the states, \f$\mathcal{A}\f$ are the actions, \f$\mathcal{O}\f$ are the outcomes.
            
    The states, actions, and outcomes are identified by consecutive ids, independently
    numbered for each type.
    
    Initialization requires the number of states.
    
    Parameters
    ----------
    statecount : int, optional
        An estimate of the numeber of states (for pre-allocation). When more states
        are added, the estimate is readjusted.
    discount : double, optional
        The discount factor
    """
    
    cdef shared_ptr[CRMDP] thisptr
    cdef public double discount

    def __cinit__(self, long statecount, double discount):
        self.thisptr = make_shared[CRMDP](statecount)

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
        add_transition[CRMDP](dereference(self.thisptr),fromid, actionid, outcomeid,
                                toid, probability, reward)

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

    cpdef long transition_count(self, long stateid, long actionid, long outcomeid):
        """
        Number of transitions (sparse transition probability) following a state,
        action, and outcome

        Parameters
        ----------
        stateid : int
            State index
        actionid : int
            Action index
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).size()

    cpdef double get_reward(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ 
        Returns the reward for the given state, action, and outcome 

        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_reward(sampleid)
        
    cpdef get_rewards(self, long stateid, long actionid, long outcomeid):
        """ 
        Returns the reward for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_rewards()

    cpdef long get_toid(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ 
        Returns the target state for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_indices()[sampleid]
        
    cpdef get_toids(self, long stateid, long actionid, long outcomeid):
        """ 
        Returns the target state for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_indices()

    cpdef double get_probability(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ 
        Returns the probability for the given state, action, outcome, and index of a non-zero transition probability
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_probabilities()[sampleid]
    
    cpdef get_probabilities(self, long stateid, long actionid, long outcomeid):
        """ 
        Returns the list of probabilities for the given state, action, and outcome 
        
        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        """
        return dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).get_probabilities()

    cpdef set_reward(self, long stateid, long actionid, long outcomeid, long sampleid, double reward):
        """
        Sets the reward for the given state, action, outcome, and sample

        Parameters
        ----------
        stateid : int
            Originating state
        actionid : int
            Action taken
        outcomeid : int
            Uncertain outcome (robustness)
        sampleid : int
            Index of the "sample" used in the sparse representation of the transition probabilities
        reward : double 
            New reward
        """
        dereference(self.thisptr).get_state(stateid).get_action(actionid).get_outcome(outcomeid).set_reward(sampleid, reward)

    cpdef set_distribution(self, long fromid, long actionid, np.ndarray[double] distribution):
        """
        Sets the base distribution over outcomes
        
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

        set_outcome_dst[CRMDP](dereference(self.thisptr), fromid, actionid, distribution)
        
    cpdef set_uniform_distributions(self):
        """ Sets all the outcome distributions to be uniform. """
        set_uniform_outcome_dst[CRMDP](dereference(self.thisptr))

    cpdef copy(self):
        """ Makes a copy of the object """
        r = RMDP(0, self.discount)
        r.thisptr.reset(new CRMDP(dereference(self.thisptr)))
        return r

    cpdef robustify_mdp(self, MDP mdp, bool allowzeros):
        """
        Overwrites current RMDP with a robustified version of the 
        provided MDP. 

        If allowzeros = True the there is an outcome k for every state k and the
        transition from outcome k is directly to state k (deterministic). This way,
        even the probability of transitioning to any state 0, the realization of 
        the robust uncertainty may have non-zero weight for that state.

        If allowzeros = False then outcomes are added only for transitions with positive
        probabilities.

        The initial thresholds are all set to 0.

        When allowzeros = false, then robust solutions to the RMDP should be the same 
        as the robust solutions to the MDP.

        Parameters
        ----------
        mdp : MDP   
            The source MDP
        allowzeros : bool
            Whether to allow outcomes to states with zero transition probabilities
        """
        cdef CRMDP rmdp = robustify(dereference(mdp.thisptr), allowzeros)
        self.thisptr.reset(new CRMDP(rmdp))

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
        cdef long actioncount = len(actions) # really the number of action
        cdef long statecount = transitions.shape[0]

        if actioncount != transitions.shape[2] or actioncount != rewards.shape[1]:
            raise ValueError('The number of actions must match the 3rd dimension of transitions and the 2nd dimension of rewards.')
        if statecount != transitions.shape[1] or statecount != rewards.shape[0]:
            raise ValueError('The number of states in transitions and rewards is inconsistent.')
        if len(set(actions)) != actioncount:
            raise ValueError('The actions must be unique.')

        cdef long aoindex, fromid, toid
        cdef long actionid 
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

    cpdef solve_vi(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                            policy = np.empty(0), double maxresidual=0):
        """
        Runs regular value iteration (no robustness or nature response).
        
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

        cdef Solution sol = csolve_vi_rmdp(dereference(self.thisptr), self.discount,\
                    valuefunction,policy,iterations,maxresidual)

        return SolutionTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations)


    cpdef solve_mpi(self, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                                    policy = np.empty(0),
                                    double maxresidual = 0, long valiterations = -1,  
                                    double valresidual=-1, bool show_progress = False):
        """
        Runs regular modified policy iteration with parallelized Jacobi valus updates. No robustness
        or nature response.
        
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

        cdef Solution sol = csolve_mpi_rmdp(dereference(self.thisptr),self.discount,\
                        valuefunction,policy,iterations,maxresidual,valiterations,\
                        valresidual,show_progress)

        return SolutionTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations)

    cpdef rsolve_vi(self, nature, thresholds, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                            policy = np.empty(0), double maxresidual=0):
        """
        Runs robust value iteration.
        
        This is the "Gauss-Seidel" kind of value iteration in which the state values
        are updated one at a time and directly used in subsequent iterations.
        
        This version is not parallelized (and likely would be hard to).

        Returns a namedtuple SolutionRobustTuple.
        
        Parameters
        ----------
        nature : string
            Type of response of nature. See choose_nature for supported values.
        thresholds : (stateids, actionids, thresholds values)
            Each entry represents the threshold for each state and action. The threshold
            should be provided for each state and action value; the ones that are not 
            specified are undefined.
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
        natpolicy : np.ndarray
            Best responses selected by nature
        """
        
        self._check_value(valuefunction)
        self._check_policy(policy)

        cdef SolutionRobust sol = crsolve_vi(dereference(self.thisptr), self.discount,\
                        string_to_nature(nature), pack_thresholds(thresholds[0], thresholds[1], thresholds[2]),
                        valuefunction,policy,iterations,maxresidual)

        return SolutionRobustTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations,sol.natpolicy)


    cpdef rsolve_mpi(self, nature, thresholds, long iterations=DEFAULT_ITERS, valuefunction = np.empty(0),
                                    policy = np.empty(0),
                                    double maxresidual = 0, long valiterations = -1, 
                                    double valresidual=-1, bool show_progress = False):
        """
        Runs robust modified policy iteration with parallelized Jacobi value updates.         

        Returns a namedtuple SolutionRobustTuple.

        Parameters
        ----------
        nature : string
            Type of response of nature. See choose_nature for supported values.
        thresholds : (stateids, actionids, thresholds values)
            Each entry represents the threshold for each state and action. The threshold
            should be provided for each state and action value; the ones that are not 
            specified are undefined.
        natpolicy : np.ndarray
            Best responses selected by nature
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
        natpolicy : np.ndarray
            Best responses selected by nature
        """

        self._check_value(valuefunction)
        self._check_policy(policy)

        if valiterations <= 0: valiterations = iterations
        if valresidual < 0: valresidual = maxresidual / 2

        cdef SolutionRobust sol = crsolve_mpi(dereference(self.thisptr),self.discount,\
                        string_to_nature(nature), pack_thresholds(thresholds[0], thresholds[1], thresholds[2]),
                        valuefunction,policy,iterations,maxresidual,valiterations,\
                        valresidual,show_progress)

        return SolutionRobustTuple(np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.natpolicy)



# ***************************************************************************
# *******    Implementable    *******
# ***************************************************************************

cdef extern from "craam/ImMDP.hpp" namespace 'craam::impl':
    
    cdef cppclass MDPI_R:
    
        MDPI_R(const CMDP& mdp, const indvec& observ2state, const CTransition& initial);

        vector[long] obspol2statepol(const vector[long]& obspol) except +;
        
        const CRMDP& get_robust_mdp() except +

        vector[long] solve_reweighted(long iterations, double discount) except +;
        vector[long] solve_robust(long iterations, double threshold, double discount) except +;
        
        double total_return(double discount, double precision);
        
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
        result.thisptr = make_shared[CRMDP](dereference(self.thisptr).get_robust_mdp())
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
        """ The return of an interpretable policy """
        assert len(obspol) == self.obs_count()
        return dereference(self.thisptr).total_return(self.discount, 1e-8)

    def to_csv(self, mdp_file, state2obs_file, initial_file, headers):
        """
        Saves the problem to a csv file
        """
        dereference(self.thisptr).to_csv_file(mdp_file, state2obs_file, initial_file, headers)

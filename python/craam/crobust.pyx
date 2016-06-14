# distutils: language = c++
# distutils: libraries = craam
# distutils: library_dirs = craam/lib 
# distutils: include_dirs = ../../craam/include

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

cdef extern from "../../craam/include/RMDP.hpp" namespace 'craam':
    pair[vector[double],double] worstcase_l1(const vector[double] & z, \
                        const vector[double] & q, double t)
                                            
    cdef cppclass Solution:
        vector[double] valuefunction
        vector[long] policy
        vector[long] outcomes
        vector[vector[double]] natpolicy
        double residual
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

    cdef cppclass Action:
        
        void set_threshold(double threshold) except +
        void set_distribution(const vector[double]& distribution) except +
        void set_distribution(long outcomeid, double weight) except + 
        
        Transition& get_transition(long outcomeid) 
        
        size_t outcome_count()
        double get_threshold()

        Transition& get_outcome(long outcomeid) except +

    cdef cppclass State:
        Action& get_action(long actionid) 
        size_t action_count() except +

    cdef cppclass RMDP:

        RMDP(int state_count) except +

        void add_transition(long fromid, long actionid, long outcomeid, 
                            long toid, double probability, double reward) except + 

        void set_uniform_distribution(double threshold)
        void set_uniform_thresholds(double threshold) except +

        State& get_state(long stateid) 

        void normalize()

        long state_count() except +
        long transitions_count(long stateid, long actionid, long outcomeid) except +
        
        Transition get_transition(long stateid,long actionid,long outcomeid) except +

        Solution vi_gs_rob(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        Solution vi_gs_opt(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        Solution vi_gs_ave(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        
        Solution vi_gs_l1_rob(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        Solution vi_gs_l1_opt(vector[double] valuefunction, double discount, unsigned long iterations, double maxresidual) except +

        Solution vi_jac_rob(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        Solution vi_jac_opt(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        Solution vi_jac_ave(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        
        Solution vi_jac_l1_rob(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual) except +
        Solution vi_jac_l1_opt(const vector[double] & valuefunction, double discount, unsigned long iterations, double maxresidual) except +

        Solution mpi_jac_rob(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual) except +
        Solution mpi_jac_opt(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual) except +
        Solution mpi_jac_ave(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual) except +
        
        Solution mpi_jac_l1_rob(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual) except +
        Solution mpi_jac_l1_opt(const vector[double] & valuefunction, double discount, unsigned long politerations, double polmaxresidual, unsigned long valiterations, double valmaxresidual) except +

        void to_csv_file(const string & filename, bool header) except +
        string to_string() except +


cdef extern from "../../craam/include/Samples.hpp" namespace 'craam::msen':
    
    cdef cppclass DiscreteESample:
    
        const long expstate_from;
        const long decstate_to;
        const double reward;
        const double weight;
        const long step;
        const long run;
    
        DiscreteESample(const long& expstate_from, const long& decstate_to,
                double reward, double weight, long step, long run);
    
    cdef cppclass DiscreteDSample:
    
        const long decstate_from;
        const long action;
        const long expstate_to;
        const long step;
        const long run;
    
        DiscreteDSample(const long& decstate_from, const long& action,
                const long& expstate_to, long step, long run);

    cdef cppclass DiscreteSamples:
        vector[DiscreteDSample] decsamples;
        vector[long] initial;
        vector[DiscreteESample] expsamples;
    
        void add_dec(const DiscreteDSample& decsample);
        void add_initial(const long& decstate);
        void add_exp(const DiscreteESample& expsample);
        double mean_return(double discount);
        
        DiscreteSamples();

    cdef cppclass SampledMDP:
        
        SampledMDP()
    
        void add_samples(const DiscreteSamples& samples) except +
        shared_ptr[const RMDP] get_mdp() except +
        Transition get_initial() except +


cdef extern from "../../craam/include/ImMDP.hpp" namespace 'craam::impl':
    
    cdef cppclass MDPI_R:
    
        MDPI_R(const RMDP& mdp, const vector[long]& observ2state, const Transition& initial);

        vector[long] obspol2statepol(const vector[long]& obspol) except +;
        
        const RMDP& get_robust_mdp() except +

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


# a contained used to hold the dictionaries used to map sample states to MDP states
StateMaps = namedtuple('StateMaps',['decstate2state','expstate2state',\
                                    'decstate2outcome','expstate2outcome'])

cdef class RoMDP:
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
    
    cdef RMDP *thisptr
    cdef public double discount

    def __cinit__(self, int statecount, double discount):
        self.thisptr = new RMDP(statecount)

    def __init__(self, int statecount, double discount):
        self.discount = discount
        
    def __dealloc__(self):
        del self.thisptr
                
    cpdef add_transition(self, long fromid, long actionid, long outcomeid, long toid, double probability, double reward):
        """
        Adds a single transition sample (robust or non-robust) to the Robust MDP representation.
        
        Parameters
        ----------
        fromid : long
            Unique identifier of the source state of the transition 
        actionid : long
            Identifier of the action. It is unique for the given state
        outcomeid : long
            Identifier of the outcome. It is unique for the given state
        toid : long
            Unique identifier of the target state of the transition
        probability : float
            Probability of the distribution
        reward : float
            Reward associated with the transition
        """
        self.thisptr.add_transition(fromid, actionid, outcomeid, toid, probability, reward)

    cpdef add_transition_d(self, long fromid, long actionid, long toid, double probability, double reward):
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
        self.thisptr.add_transition(fromid, actionid, 0, toid, probability, reward)

    
    cpdef add_transition_nonrobust(self, long fromid, long actionid, long toid, double probability, double reward):
        """
        Deprecated, use add_transition_d
        """
        self.thisptr.add_transition(fromid, actionid, 0, toid, probability, reward)


    cpdef set_distribution(self, long fromid, long actionid, np.ndarray[double] distribution, double threshold):
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
        threshold : double
            The difference threshold used when choosing the outcomes
        """
        if abs(np.sum(distribution) - 1) > 0.001:
            raise ValueError('incorrect distribution (does not sum to one)', distribution)
        if np.min(distribution) < 0:
            raise ValueError('incorrect distribution (negative)', distribution)    

        self.thisptr.get_state(fromid).get_action(actionid).set_distribution(distribution)
        self.thisptr.get_state(fromid).get_action(actionid).set_threshold(threshold)
        
    cpdef set_uniform_distributions(self, double threshold):
        """
        Sets all the outcome distributions to be uniform.
        
        Parameters
        ----------
        threshold : double
            The default threshold for the uncertain sets
        """
        self.thisptr.set_uniform_distribution(threshold)

    cpdef set_uniform_thresholds(self, double threshold):
        """
        Sets the same threshold for all states.
        
        Can use ``self.set_distribution`` to set the thresholds individually for 
        each states and action.
        
        See Also
        --------
        self.set_distribution
        """
        self.thisptr.set_uniform_thresholds(threshold)

    cpdef vi_gs(self, int iterations, valuefunction = np.empty(0), \
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
            can use e.g. robust.SolutionType.Robust.value.
            
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
        
        See Also
        --------
        SolutionType
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        cdef Solution sol

        if stype == 0:
            sol = self.thisptr.vi_gs_rob(valuefunction,self.discount,iterations,maxresidual)
        elif stype == 1:
            sol = self.thisptr.vi_gs_opt(valuefunction,self.discount,iterations,maxresidual)
        elif stype == 2:
            sol = self.thisptr.vi_gs_ave(valuefunction,self.discount,iterations,maxresidual)
        else:
            raise ValueError("Incorrect solution type '%s'." % stype )
 
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes

    cpdef vi_gs_l1(self, int iterations, valuefunction = np.empty(0), \
                    double maxresidual=0, int stype=0):
        """
        Runs value iteration using the worst distribution constrained by the threshold 
        and l1 norm difference from the base distribution.
        
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
        stype : int  {0, 1}, optional
            Robust (0) or optimistic (1) solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
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
        natpolicy : np.ndarray[np.ndarray]
            Distributions of outcomes
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        cdef Solution sol
        
        if stype == 0:
            sol = self.thisptr.vi_gs_l1_rob(valuefunction,self.discount,iterations,maxresidual)
        elif stype == 1:
            sol = self.thisptr.vi_gs_l1_opt(valuefunction,self.discount,iterations,maxresidual)
        else:
            raise ValueError("Incorrect solution type '%s'."  % stype )
            
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.natpolicy
        
    cpdef vi_jac(self, int iterations,valuefunction = np.empty(0), \
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
            can use e.g. robust.SolutionType.Robust.value.
            
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
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')        
                
        cdef Solution sol
        if stype == 0:
            sol = self.thisptr.vi_jac_rob(valuefunction,self.discount,iterations,maxresidual)
        elif stype == 1:
            sol = self.thisptr.vi_jac_opt(valuefunction,self.discount,iterations,maxresidual)
        elif stype == 2:
            sol = self.thisptr.vi_jac_ave(valuefunction,self.discount,iterations,maxresidual)
        else:
            raise ValueError("Incorrect solution type '%s'."  % stype )
            
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes

    cpdef vi_jac_l1(self, long iterations, valuefunction = np.empty(0), \
                                    double maxresidual = 0, int stype=0):
        """
        Runs value iteration using the worst distribution constrained by the threshold 
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
        stype : int  (0, 1}
            Robust (0) or optimistic (1) solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
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
        natpolicy : np.ndarray[np.ndarray]
            Distributions of outcomes
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        cdef Solution sol
        if stype == 0:
            sol = self.thisptr.vi_jac_l1_rob(valuefunction,self.discount,iterations,maxresidual)
        elif stype == 1:
            sol = self.thisptr.vi_jac_l1_opt(valuefunction,self.discount,iterations,maxresidual)
        else:
            raise ValueError("Incorrect solution type '%s'."  % stype )
        
        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.natpolicy

    cpdef mpi_jac(self, long iterations, valuefunction = np.empty(0), \
                                    double maxresidual = 0, long valiterations = 1000, int stype=0):
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
            can use e.g. robust.SolutionType.Robust.value.
            
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
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        # TODO: what it the best value to use here?
        cdef double valresidual = maxresidual / 2

        cdef Solution sol
        if stype == 0:
            sol = self.thisptr.mpi_jac_rob(valuefunction,self.discount,iterations,maxresidual,valiterations,valresidual)
        elif stype == 1:
            sol = self.thisptr.mpi_jac_opt(valuefunction,self.discount,iterations,maxresidual,valiterations,valresidual)
        elif stype == 2:
            sol = self.thisptr.mpi_jac_ave(valuefunction,self.discount,iterations,maxresidual,valiterations,valresidual)
        else:
            raise ValueError("Incorrect solution type '%s'."  % stype )            

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual, \
                sol.iterations, sol.outcomes


    cpdef mpi_jac_l1(self, long iterations, valuefunction = np.empty(0), \
                        long valiterations = 1000, double maxresidual = 0, int stype=0):
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
        stype : int  (0, 1}
            Robust (0) or optimistic (1) solution. One
            can use e.g. robust.SolutionType.Robust.value.
            
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
        natpolicy : np.ndarray[np.ndarray]
            Distributions of outcomes
        """
        
        if valuefunction.shape[0] == 0:
            # create an appropriate zero initial value function vector
            valuefunction = np.zeros(self.thisptr.state_count(), dtype=np.double)
        elif valuefunction.shape[0] != self.thisptr.state_count():
            raise ValueError('Value function dimensions must match the number of states.')

        # TODO: what it the best value to use here?
        cdef double valresidual = maxresidual / 2

        cdef Solution sol
        if stype == 0:
            sol = self.thisptr.mpi_jac_l1_rob(valuefunction,self.discount,iterations,maxresidual,valiterations, valresidual)
        elif stype == 1:
            sol = self.thisptr.mpi_jac_l1_opt(valuefunction,self.discount,iterations,maxresidual,valiterations, valresidual)
        else:
            raise ValueError("Incorrect solution type '%s'."  % stype )            

        return np.array(sol.valuefunction), np.array(sol.policy), sol.residual,\
                sol.iterations, sol.natpolicy

    cpdef from_sample_matrices(self, dectoexp, exptodec, actions, rewards):
        """
        Add samples from matrices generated by raam.robust.matrices.

        Note: The base distributions over the outcomes are assumed to be uniform.
        
        Parameters
        ----------
        dectoexp : numpy.ndarray
            List of transitions for all aggregate states. For each aggregate state
            the list contains a *masked* matrix (np.ndarray) with dimensions 
            #actions x #outcomes. The entries with no samples are considered to be masked.
            Each outcome corresponds to a decision state (from decision samples)
            with a unique index. Each entry is an index of the corresponding
            expectation state (row number in exptodec).
        exptodec : scipy.sparse.dok_matrix
            A sparse transition matrix from expectation states to decision states.
        actions : list
            List of actions available in the problem. Actions are sorted alphabetically.
        rewards : numpy.ndarray
            Average reward for each expectation state
        """
        cdef int actioncount = len(actions)
        cdef int statecount = dectoexp.shape[0]
        cdef int s,a,o,ns

        for s in range(statecount):
            actionoutcomes = dectoexp[s]
            for a in range(actioncount):
                if actionoutcomes is None:
                    continue
                outcomecount = actionoutcomes.shape[1]
                realoutcomecount = 0
                for o in range(outcomecount):
                    if actionoutcomes.mask[a,o]:
                        continue
                    es = exptodec[actionoutcomes[a,o],:].tocoo()
                    rew = rewards[actionoutcomes[a,o]]
                    
                    es_size = es.col.shape[0]
                    cols = es.col
                    data = es.data
                    
                    for ns in range(es_size):
                        toid = cols[ns]
                        prob = data[ns]
                        self.add_transition(s,a,realoutcomecount,toid,prob,rew)
                    realoutcomecount += 1
                
                if(realoutcomecount > 0):
                    dist = np.ones(realoutcomecount) / realoutcomecount
                    self.set_distribution(s,a,dist,2)

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
            The last dimension represents the actions and outcomes as defined by
            the parameters `action` and `outcomes'. The first dimension represents
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
        cdef int actioncount = len(actions) # really the number of action-outcome pairs
        cdef int statecount = transitions.shape[0]

        if actioncount != len(outcomes):
            raise ValueError('Length of actions and outcomes must match.')
        if actioncount != transitions.shape[2] or actioncount != rewards.shape[1]:
            raise ValueError('The number of actions must match the 3rd dimension of transitions and the 2nd dimension of rewards.')
        if statecount != transitions.shape[1] or statecount != rewards.shape[0]:
            raise ValueError('The number of states in transitions and rewards is inconsistent.')
        if len(set(zip(actions,outcomes))) != actioncount:
            raise ValueError('The action and outcome pairs must be unique.')

        cdef int aoindex, fromid, toid
        cdef int actionid, outcomeid
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


    cpdef long state_count(self):
        """
        Returns the number of states
        """
        return self.thisptr.state_count()
        
    cpdef long action_count(self, long stateid):
        """
        Returns the number of actions
        
        Parameters
        ----------
        stateid : int
            Number of the state
        """
        return self.thisptr.get_state(stateid).action_count()
        
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
        return self.thisptr.get_state(stateid).get_action(actionid).outcome_count()

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
        return self.thisptr.get_state(stateid).get_action(actionid).get_outcome(outcomeid).size()


    cpdef double get_reward(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ Returns the reward for the given state, action, and outcome """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).get_reward(sampleid)
        
    cpdef get_rewards(self, long stateid, long actionid, long outcomeid):
        """ Returns the reward for the given state, action, and outcome """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).get_rewards()

    cpdef long get_toid(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ Returns the target state for the given state, action, and outcome """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).get_indices()[sampleid]
        
    cpdef get_toids(self, long stateid, long actionid, long outcomeid):
        """ Returns the target state for the given state, action, and outcome """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).get_indices()

    cpdef double get_probability(self, long stateid, long actionid, long outcomeid, long sampleid):
        """ Returns the probability for the given state, action, and outcome """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).get_probabilities()[sampleid]
    
    cpdef get_probabilities(self, long stateid, long actionid, long outcomeid):
        """ Returns the probability for the given state, action, and outcome """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).get_probabilities()

    cpdef set_reward(self, long stateid, long actionid, long outcomeid, long sampleid, double reward):
        """
        Sets the reward for the given state, action, outcome, and sample
        """
        self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).set_reward(sampleid, reward)
        
    cpdef long sample_count(self, long stateid, long actionid, long outcomeid):
        """
        Returns the number of samples (single-state transitions) for the action and outcome
        """
        return self.thisptr.get_state(stateid).get_action(actionid).get_transition(outcomeid).size()

    def list_samples(self):
        """
        Returns a list of all samples in the problem. Can be useful for debugging.
        """
        cdef long stateid, actionid, outcomeid
        cdef Transition tran
        cdef long tcount, tid
        
        result = [('state','action','outcome','tostate','prob','rew')]
        
        for stateid in range(self.state_count()):
            for actionid in range(self.action_count(stateid)):
                for outcomeid in range(self.outcome_count(stateid,actionid)):
                    tran = self.thisptr.get_transition(stateid,actionid,outcomeid)
                    tcount = tran.size()
                    for tid in range(tcount):
                        result.append( (stateid,actionid,outcomeid,\
                            tran.get_indices()[tid], tran.get_probabilities()[tid], \
                            tran.get_rewards()[tid]) )

        return result

    cpdef copy(self):
        """
        Makes a copy of the object
        """
        r = RoMDP(0, self.discount)
        r.thisptr[0] = self.thisptr[0]
        return r

    cpdef double get_threshold(self, long stateid, long actionid):
        """ Returns the robust threshold for a given state """
        return self.thisptr.get_state(stateid).get_action(actionid).get_threshold()
    
    cpdef set_threshold(self, long stateid, long actionid, double threshold):
        """ Sets the robust threshold for a given state """
        self.thisptr.get_state(stateid).get_action(actionid).set_threshold(threshold)
        
    cpdef to_csv_file(self, filename, header = True):
        """ Saves the transitions to a csv file """
        self.thisptr.to_csv_file(filename, header)

    cpdef string to_string(self):
        cdef string result = self.thisptr.to_string()
        return result 

    cpdef normalize(self):
        self.thisptr.normalize()

    def __dealloc__(self):
        del self.thisptr


class SRoMDP:
    """
    Robust MDP constructed from samples and an aggregation.
    
    See :method:`from_samples` for the description of basic usage.
    
    Important: the discount factor used internally with the RoMDP must 
    be sqrt(discount) for to behave as discount; this correction is handled 
    automatically by the class.

    Parameters
    ----------
    states : int
        Initial number of states. State space is automatically expanded when more
        samples become available.
    discount : float
        Discount factor used in the MDP.
    """

    def __init__(self,states,discount):
        # state mappings
        self.rmdp = RoMDP(states,sqrt(discount))
        self.discount = discount
        
        # decstate2state, expstate2state, decstate2outcome, expstate2outcome
        self.statemaps = StateMaps({},{},{},{})
        
        # the following dictionaries are used in order to properly weigh samples 
        # when added multiple times
                        
        # decision states
        self.dcount_sao = {} # maps state,action,outcome to the number of observations
            # this is uses to combine multiple samples
        
        # expectation states
        self.ecount_sao = {} # maps state,action,outcome to the number of observations
            # this is uses to combine multiple samples

        # initial distribution; maps states to probabilities
        self.initial_distribution = {}
    
    def from_samples(self, samples, decagg_big, decagg_small, \
                        expagg_big, expagg_small, actagg):
        """
        Loads samples to the MDP from the provided samples given aggregation functions.
        Each decision state that belongs to a single aggregated state corresponds 
        to an (worst-case) outcome. The function does not return anything.
        
        Both expectation and decision states are translated to separate RMDP states
        
        Important: the discount factor used internally with the RoMDP must 
        be sqrt(discount) for to behave as discount; this correction is handled 
        automatically by the class.

        The method also estimates the initial distribution from the samples
        
        This method can only be called once for an object

        Parameters
        ----------
        samples : raam.Samples
            List of samples
        decagg_big : function
            Aggregation function for decision states, used to construct the 
            actual aggregation. The function should return an integer 
            (could be negative).
        decagg_small : function
            Aggregation function used to construct outcomes and the value is 
            relative to the state given by ``decagg_big``. The solution is 
            computed as the worst-case over these outcomes. This can be just a
            finer aggregation function than decagg_big, or could come from 
            multiple runs. The function should return an integer 
            (could be negative). Use ``features.IdCache`` to simply use the state
            identity.
        expagg_big : function
            Aggregation function for expectation states. This is used to average
            the transition probabilities. The function should return an integer 
            (could be negative). Use features.IdCache to simply use the state
            identity.
        expagg_small : function
            Aggregation function used to construct outcomes for expectation states.
            The function can be ``None`` which means that all expectation states
            are aggregated into one.
        actagg : function
            Aggregation function for actions. The function should return an integer 
            (could be negative).
        
        Note
        ----
        If the aggregation functions return a floating point then the number is
        rounded to an integer and used as the index of the state or the action. 
        
        See Also
        --------
        decvalue
        """
        cdef long aggds_big, aggds_small, agges_big, agges_small
        cdef long numdecstate, numexpstate, numaction, numoutcome
        cdef long mdpstates = self.rmdp.state_count()
        cdef RoMDP rmdp = self.rmdp        
        
        # if there is no small aggregation provided, then just assume that there 
        # is no aggregation
        if expagg_small is None:
            expagg_small = lambda x: 0
        
        # maps decision states (aggregated) to the states of the RMDP
        decstate2state = self.statemaps.decstate2state
        # maps expectation states (aggregated) to the states of the RMDP
        expstate2state = self.statemaps.expstate2state
        # maps decision states (small aggregation) to dictionaries of outcomes in the MDP
        decstate2outcome = self.statemaps.decstate2outcome
        # maps expectation states (small aggregated) to  outcomes in the MDP
        expstate2outcome = self.statemaps.expstate2outcome
        
        dcount_sao_old = self.dcount_sao.copy()
        ecount_sao_old = self.ecount_sao.copy()
        
        # *** process decision samples
        for ds in samples.decsamples():
            
            # compute the mdp state for the decision state
            aggds_big = decagg_big(ds.decStateFrom)
            if aggds_big in decstate2state:
                numdecstate = decstate2state[aggds_big]
            else:
                numdecstate = mdpstates
                mdpstates += 1
                decstate2state[aggds_big] = numdecstate
            
            # compute the mdp state for the expectation state
            agges_big = expagg_big(ds.expStateTo)
            if agges_big in expstate2state:
                numexpstate = expstate2state[agges_big]
            else:
                numexpstate = mdpstates
                mdpstates += 1
                expstate2state[agges_big] = numexpstate
            
            # compute action aggregation, the mapping is 1->1
            numaction = actagg(ds.action)
            
            # compute the outcome aggregation
            aggds_small = decagg_small(ds.decStateFrom)
            
            outcomedict = decstate2outcome.get((aggds_big,numaction), None)
            if outcomedict is None:
                outcomedict = {}
                decstate2outcome[(aggds_big,numaction)] = outcomedict
            
            if aggds_small in outcomedict:
                numoutcome = outcomedict[aggds_small]
            else:
                numoutcome = len(outcomedict)
                outcomedict[aggds_small] = numoutcome

            # update the counts for the sample
            sao = (numdecstate,numaction,numoutcome)
            self.dcount_sao[sao] = self.dcount_sao.get(sao,0) + 1
            
            # now, just add the transition
            # use the old counts to compute the weight in order for the normalization
            # to work
            weight = 1.0 / float(dcount_sao_old.get(sao,1))
            rmdp.add_transition(numdecstate,numaction,numoutcome,numexpstate,weight,0.0)
            
        # *** process expectation samples
        # one action and outcome per state
        for es in samples.expsamples():
            # compute the mdp state for the expectation state
            agges_big = expagg_big(es.expStateFrom)
            if agges_big in expstate2state:
                numexpstate = expstate2state[agges_big]
            else:
                numexpstate = mdpstates
                mdpstates += 1
                expstate2state[agges_big] = numexpstate

            # compute the mdp state for the decision state
            aggds_big = decagg_big(es.decStateTo)
            if aggds_big in decstate2state:
                numdecstate = decstate2state[aggds_big]
            else:
                numdecstate = mdpstates
                mdpstates += 1
                decstate2state[aggds_big] = numdecstate
            
            # compute action aggregation
            numaction = 0   # only one action
            
            # compute the outcome aggregation
            agges_small = expagg_small(es.expStateFrom)
            
            outcomedict = expstate2outcome.get(agges_big, None)
            
            if outcomedict is None:
                outcomedict = {}
                expstate2outcome[agges_big] = outcomedict
                
            if agges_small in outcomedict:
                numoutcome = outcomedict[agges_small]
            else:
                numoutcome = len(outcomedict)
                outcomedict[agges_small] = numoutcome

            # update the counts for the sample
            so = (numexpstate,numoutcome)
            self.ecount_sao[so] = self.ecount_sao.get(so,0) + 1
            
            # now, just add the transition
            # use the old counts to compute the weight in order for the normalization
            # to work
            weight = 1.0 / float(ecount_sao_old.get(so,1))    
                
            # now, just add the transition
            self.rmdp.add_transition(numexpstate,numaction,numoutcome,numdecstate,\
                            weight*es.weight,es.reward)
            
        # add a transition to a bad state for state-actions with no outcomes
        # these state-action pairs are created automatically 
        cdef double bad = -float('inf')
        cdef long numbadstate = self.rmdp.state_count()
        self.rmdp.add_transition(numbadstate,0,0,numbadstate,1.0,0)
        
        for numstate in range(self.rmdp.state_count()):
            for numaction in range(self.rmdp.action_count(numstate)):
                if self.rmdp.outcome_count(numstate,numaction) == 0:
                    self.rmdp.add_transition(numstate,numaction,0,numbadstate,1.0,bad)
        
        # normalize transition weights
        self.rmdp.normalize()

        # process initial distribution
        for dstate in samples.initialsamples():
            # compute the mdp state for the decision state
            aggds_big = decagg_big(dstate)
            if aggds_big in decstate2state:
                numdecstate = decstate2state[aggds_big]
            else:
                numdecstate = mdpstates
                mdpstates += 1
                decstate2state[aggds_big] = numdecstate
            
            if numdecstate in self.initial_distribution:
                self.initial_distribution[numdecstate] += 1
            else:
                self.initial_distribution[numdecstate] = 1

        # normalize initial distribution
        samplecount = sum(self.initial_distribution.values())
        for k,v in self.initial_distribution.items():
            self.initial_distribution[k] = v/samplecount


    def decvalue(self,states,value,minstate=0):
        """
        Corrects the value function and maps the result from an algorithm
        to the value function for decision states.
        
        The function also corrects for the discrepancy in the discount factor,
        which is applied twice. 
        
        Parameters
        ----------
        states : int
            Number of states for which the express the value function. The states
            must be numbered from 0 to states - 1
        value : numpy.array
            Value function array as an output from the optimization methods. This
            uses the internal state representation.
        minstate : int, optional
            The minimal index of a state to output. The default is 0.
        
        Returns
        -------
        out : numpy.array
            Value function for decision states
        """
        cdef long i, index
        cdef double discountadj = 1/sqrt(self.discount)
        result = np.empty((states,))
        for i in range(minstate,minstate+states):
            index = self.statemaps.decstate2state.get(i,-1)
            if index >= 0:
                result[minstate+i] = value[index] * discountadj
            else:
                result[minstate+i] = float('nan')
        return result

    def expvalue(self,states,value,minstate=0):
        """
        Corrects the value function and maps the result from an algorithm
        to the value function for expectation states.
        
        Parameters
        ----------
        states : int
            Number of states for which the express the value function. The states
            must be numbered from 0 to states - 1
        value : numpy.array
            Value function array as an output from the optimization methods. This
            uses the state internal representation.
        minstate : int, optional
            The minimal index of the state. The default is 0.
        
        Returns
        -------
        out : numpy.array
            Value function for decision states
        """
        cdef long i, index
        result = np.empty((states,))
        for i in range(minstate,minstate+states):
            index = self.statemaps.expstate2state.get(i,-1)
            if index >= 0:
                result[minstate+i] = value[index] 
            else:
                result[minstate+i] = float('nan')
        return result
                
    def decpolicy(self,states,policy,minstate=0):
        """
        Corrects the policy function (a vector) and maps the result from an algorithm
        to the policy over decision states.
        
        The function also corrects for the discrepancy in the discount factor,
        which is applied twice. 
        
        Parameters
        ----------
        states : int
            Number of states for which the express the policy function. The states
            must be numbered from 0 to states - 1
        policy : numpy.array
            Value function array as an output from the optimization methods. This
            uses an internal representation.
        minstate : int, optional
            The minimal index of the state. The default is 0.

        Returns
        -------
        out : numpy.array(int)
            Policy function for decision states
        """
        cdef long i, index
        result = np.empty((states,),dtype=int)
        for i in range(minstate,minstate+states):
            index = self.statemaps.decstate2state.get(i,-1)
            if index >= 0:
                result[minstate+i] = policy[index]
            else:
                result[minstate+i] = -1
        return result

    def statemaps(self):
        """
        Returns the maps from sample states to actual MDP states
        """
        return self.statemaps        
        
    def samplecount(self):
        """
        Returns the number of samples in the object
        """
        return sum(self.dcount_sao.values()) + sum(self.ecount_sao.values())

    def expstate_numbers(self):
        """
        Returns numbers of the internal RoMDP states that correspond to the
        expectation states as well as the expectation state numbers.
        
        Returns
        -------
        expstate_original : list
            Original numbers assigned to the expectation states
        expstate_index : list        
            Index of the expectation state in the constructed RMDP
        """
        return list( zip(*self.statemaps.expstate2state.items()) )
        
    def decstate_numbers(self):
        """
        Returns numbers of the internal RoMDP states that correspond to the
        decision states as well as the decision state numbers.

        Returns
        -------
        decstate_original : list
            Original numbers assigned to the decision states
        decstate_index : list        
            Index of the decision state in the constructed RMDP
        """
        
        return list( zip(*self.statemaps.decstate2state.items()) )

    def valreturn(self, value):
        """
        Uses the internally store initial distribution to compute the
        return for the value function

        Parameters
        ----------
        value : numpy.array
            Value function array as an output from the optimization methods. This
            uses the internal state representation.
        """
        result = 0.0
        for ds,p in self.initial_distribution.items():
            result += value[ds] * p
        return result


cdef class MDPIR:
    """
    MDP with Implementability constraints. The implementability constraints
    require states within a single observation to have the same action
    chosen by the policy.

    Uses solution methods based on solving a robust MDP.

    Parameters
    ----------
    mdp : RoMDP
        Base MDP
    state2obs : np.ndarray
        Maps states to observation indexes. The observation index is 0-based
        and two states. The optimal 
    initial : np.ndarray
        The initial distribution
    """

    cdef MDPI_R *thisptr
    cdef double discount
    
    def __cinit__(self, RoMDP mdp, np.ndarray[long] state2obs, np.ndarray[double] initial):

        cdef long states = mdp.state_count()
        
        if states != state2obs.size:
            raise ValueError('The number of MDP states must equal to the size of state2obs.')
        if state2obs.size != initial.size:
            raise ValueError('Sizes of state2obs and initial must be the same.')

        # construct the initial distribution
        cdef Transition initial_t = Transition(np.arange(states),initial,np.zeros(states))

        self.thisptr = new MDPI_R((mdp.thisptr)[0], state2obs, initial_t)

    def __init__(self, RoMDP mdp, np.ndarray[long] state2obs, np.ndarray[double] initial):
        self.discount = mdp.discount

    def __dealloc__(self):
        del self.thisptr

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
        return self.thisptr.solve_reweighted(iterations, discount)

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
        return self.thisptr.solve_robust(iterations, threshold, discount)


    def get_robust(self):
        """
        Returns the robust representation of the implementable MDP
        """
        cdef RoMDP result = RoMDP(0, self.discount)
        result.thisptr[0] = self.thisptr.get_robust_mdp()
        return result

    def obspol2statepol(self, np.ndarray[long] obspol):
        """
        Converts an observation policy to a state policy
        """
        return self.thisptr.obspol2statepol(obspol);

    def state_count(self):
        """ Number of states in the MDP """
        return self.thisptr.state_count()

    def obs_count(self):
        """ Number of observations in the interpretable MDP """
        return self.thisptr.obs_count()

    def total_return(self, np.ndarray[long] obspol):
        """ """
        assert len(obspol) == self.obs_count()
        return self.thisptr.total_return(obspol, self.discount, 1e-8)

    def to_csv(self, mdp_file, state2obs_file, initial_file, headers):
        """
        Saves the problem to a csv file
        """
        self.thisptr.to_csv_file(mdp_file, state2obs_file, initial_file, headers)

from raam import samples
    
cdef class DiscreteMemSamples:
    """
    Represent samples in which decision and expectation states, actions, 
    are described by integers. It is a wrapper around the C++ representation of samples.

    Each state, action, and expectation state must have an integral value.

    Class ``features.DiscreteSampleView`` can be used as a convenient method for assigning
    state identifiers based on the equality between states.
    """
    cdef DiscreteSamples *_thisptr

    def __cinit__(self):
        self._thisptr = new DiscreteSamples() 
        # fields of the C++ class are:
        # vector[DiscreteDSample] decsamples;
        # vector[long] initial;
        # vector[DiscreteESample] expsamples;
        
     
    def __dealloc__(self):
        del self._thisptr        
        
    def __init__(self):
        """ 
        Creates empty sample dictionary and returns it.
        Can take arguments that describe the content of the samples.
        """
        pass

    def expsamples(self):
        """
        Returns an iterator over expectation samples.
        """
        
        cdef int n = self._thisptr.expsamples.size()
        
        return (samples.ExpSample(self._thisptr.expsamples[i].expstate_from, \
                        self._thisptr.expsamples[i].decstate_to, \
                        self._thisptr.expsamples[i].reward, \
                        self._thisptr.expsamples[i].weight, \
                        self._thisptr.expsamples[i].step, \
                        self._thisptr.expsamples[i].run) \
                    for i in range(n))
    
    def decsamples(self):
        """
        Returns an iterator over decision samples.
        """
        
        cdef int n = self._thisptr.decsamples.size()
        
        return (samples.DecSample(self._thisptr.decsamples[i].decstate_from, \
                        self._thisptr.decsamples[i].action, \
                        self._thisptr.decsamples[i].expstate_to, \
                        self._thisptr.decsamples[i].step, \
                        self._thisptr.decsamples[i].run) \
                    for i in range(n))
        
    def initialsamples(self):
        """
        Returns samples of initial decision states.
        """
        cdef int n = self._thisptr.initial.size()
        
        return (self._thisptr.initial[i] for i in range(n))

    def add_exp(self, expsample):
        """
        Adds an expectation sample.
        """
        
        self._thisptr.add_exp(DiscreteESample(expsample.expStateFrom, expsample.decStateTo, \
                        expsample.reward, expsample.weight, expsample.step, expsample.run))

        
    def add_dec(self, decsample):
        """
        Adds a decision sample.
        """
        self._thisptr.add_dec(DiscreteDSample(decsample.decStateFrom, decsample.action,\
                        decsample.expStateTo, decsample.step, decsample.run))
        
    def add_initial(self, decstate):
        """
        Adds an initial state.
        """
        self._thisptr.add_initial(decstate)
        
    def copy_from(self,samples):
        """
        Copies samples, which must be discrete.
        
        Parameters
        ----------
        samples : Samples
            Source of samples
        """
        
        for es in samples.expsamples():
            self.add_exp(es)
        
        for ds in samples.decsamples():
            self.add_dec(ds)
        
        for ins in samples.initialsamples():
            self.add_initial(ins)

cdef class SMDP:
    """
    An MDP that can be constructed from samples

    The class is based on DiscreteMemSamples, which
    assign an integer to each state, and action

    Unlike SRoMDP, this class does not treat expectation states as separate states.
    The representation is simpler, but also this means that the computational 
    complexity may be much greater because of repeated transition probabilities.
    """
    
    cdef SampledMDP *_thisptr
        
    def __cinit__(self):
        self._thisptr = new SampledMDP()
    
    def __dealloc__(self):
        del self._thisptr    
        
    def copy_samples(self, DiscreteMemSamples samples):
        """
        Adds samples to the MDP representation.

        At the moment, this method can only be called once.
        """
        self._thisptr.add_samples((samples._thisptr)[0])
    
    def get_mdp(self, discount):
        """ Returns constructed MDP """
        m = RoMDP(0,discount)
        m.thisptr[0] = (self._thisptr.get_mdp().get())[0]
        return m
        
    def get_initial(self):
        """ Returns constructed initial distribution """
        cdef Transition t = self._thisptr.get_initial()
        return np.array(t.probabilities_vector(self._thisptr.get_mdp().get().state_count()))
        

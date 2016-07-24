# distutils: language = c++
# distutils: libraries = craam
# distutils: library_dirs = craam/lib 
# distutils: include_dirs = ../include ../ext_include

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

cdef extern from "../include/RMDP.hpp" namespace 'craam':
                                            
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



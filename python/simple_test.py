#!/bin/python
import craam
import numpy as np

m = craam.SMDP(0,0.9)

transitions1 = np.array(\
            [[0,1,0.0], \
             [0,0,1], \
             [1,0,0]])

transitions2 = np.array(\
            [[0,0,1.0], \
             [1,0,0], \
             [0,1,0]])

transitions = np.dstack((transitions1,transitions2))

rewards = np.array([[1.0,3],[2,2],[4,1]])
actions = np.array([0,1])

m.from_matrices(transitions, rewards, actions)

v = m.mpi_jac()[0]

sim = craam.Simulation(m)

samples = sim.simulate_random();

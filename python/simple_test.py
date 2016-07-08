#!/bin/python
import craam
import numpy as np

m = craam.MDP(0,0.9)

tran_action0 = np.array(\
            [[0,0.8,0.2], \
             [0,0,1], \
             [1,0,0]])

tran_action1 = np.array(\
            [[0,0.2,0.8], \
             [1,0,0], \
             [0,1,0]])

transitions = np.dstack((tran_action0,tran_action1))

rewards = np.array([[1.0,3],[2,2],[4,1]])
actions = np.array([0,1])

m.from_matrices(transitions, rewards, actions)

v1,p1,_,_ = m.mpi_jac()

sim = craam.Simulation(m, np.array([0.1,0.5,0.4]))

samples = sim.simulate_random()

sampledMDP = craam.SampledMDP()
sampledMDP.add_samples(samples)

m2 = sampledMDP.get_mdp(m.discount)

v2,p2,_,_ = m2.mpi_jac()

print(v1,p1)
print(v2,p2)

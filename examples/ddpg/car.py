from pygent.environments import Car
from pygent.algorithms.ddpg import DDPG
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, u2 = u
    c = x1**2 + x2**2 + 10*u1**2 + 0.1*u2**2
    return c

# initial state value
x0 = [1., 1., 3./2.*np.pi, 0.]

t = 15 # simulation time
dt = 0.03 # time step-size

env = Car(c_k, x0, dt)

env.terminal_cost = 200

path = '../../../results/car/ddpg/' # path, where results are saved

algorithm = DDPG(env, t, dt, path=path) # instance of the DDPG algorithm

#algorithm.load() # can be used to load existing networks and data set

learning_steps = 5e5 # define training duration
algorithm.run_learning(learning_steps) # run reinforcment learning
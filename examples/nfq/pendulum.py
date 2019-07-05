from pygent.environments import Pendulum
from pygent.algorithms.nfq import NFQ
import numpy as np

# define the incremental cost
def c_k(x):
    x1, x2 = x
    if abs(x1) < 0.1  and abs(x2) < 0.5:
        c = 0.
    else:
        c = 0.01
    return c

# initial state value
x0 = [np.pi, 0]

t = 6 # simulation time
dt = 0.05 # time step-size

env = Pendulum(c_k, x0, dt)

controls = np.array([-5, 0, 5]).T # possible control values, the agent can choose

xGoal = [0, 0] # goal state

path = '../../../results/nfq/pendulum/' # path, where results are saved

algorithm = NFQ(env, controls, xGoal, t, dt, path=path)  # instance of the NFQ algorithm

episodes = 400 # number of episodes
algorithm.run_learning(episodes) # run reinforcement learning
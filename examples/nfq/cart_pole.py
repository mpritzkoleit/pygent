from pygent.environments import CartPole
from pygent.algorithms.nfq import NFQ
import numpy as np

# define the incremental cost
def c_k(x):
    x1, x2, x3, x4 = x
    if abs(x2) < 0.2 and abs(x1) < 0.1:
        c = 0
    else:
        c = 0.01
    return c

# initial state value
x0 = [0, np.pi, 0, 0]

t = 10 # simulation time
dt = 0.05 # time step-size

env = CartPole(c_k, x0, dt)

controls = np.array([-5, 0, 5]).T # possible control values, the agent can choose

xGoal = [0, 0, 0, 0] # goal state

path = '../../../results/nfq/cart_pole/' # path, where results are saved

algorithm = NFQ(env, controls, xGoal, t, dt, path=path)  # instance of the NFQ algorithm

episodes = 400
algorithm.run_learning(episodes)

from pygent.environments import Acrobot
from pygent.algorithms.ddpg import DDPG
import numpy as np

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = x1**2 + x2**2 + 0.01*x3**2 + 0.01*x4**2 + 0.05*u1**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(0.99*np.pi, 1.01*np.pi), np.random.uniform(-.01, .01), 0, 0]
    return x0

t = 10 # time of an episode
dt = 0.03 # time step-size

env = Acrobot(c_k, p_x0, dt)

env.terminal_cost = 200 # define the terminal cost if x(k+1) is a terminal state

path = '../../../results/acrobot/ddpg/'  # path, where results are saved

algorithm = DDPG(env, t, dt, path=path) # instance of the DDPG algorithm

#algorithm.load() # can be used to load existing networks and data set

learning_steps = 5e5 # define training duration
algorithm.run_learning(learning_steps) # run reinforcment learning
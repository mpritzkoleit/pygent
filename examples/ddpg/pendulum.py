from pygent.environments import Pendulum
from pygent.algorithms.ddpg import DDPG
import numpy as np

# define the incremental cost
def c_k(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + 0.1*x2**2 + 0.05*u1**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(0.999*np.pi, 1.001*np.pi), np.random.uniform(-0.001,0.001)]
    return x0

t = 10 # time of an episode
dt = 0.05 # time step-size

env = Pendulum(c_k, p_x0, dt)

path = '../../../results/pendulum/ddpg/'  # path, where results are saved

algorithm = DDPG(env, t, dt, path=path) # instance of the DDPG algorithm

#algorithm.load() # can be used to load existing networks and data set

learning_steps = 1e6 # define training duration
algorithm.run_learning(learning_steps) # run reinforcment learning
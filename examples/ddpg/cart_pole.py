from pygent.environments import CartPole
from pygent.algorithms.ddpg import DDPG
import numpy as np
from pygent.helpers import mapAngles
# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = 0.5*x1**2 + x2**2 + 0.02*x3**2 + 0.1*x4**2 + 0.1*u1**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.001, 0.001), np.random.uniform(0.999*np.pi, 1.001*np.pi), 0, 0]
    return x0

t = 10 # time of an episode
dt = 0.02 # time step-size

env = CartPole(c_k, p_x0, dt)

env.terminal_cost = 2000 # define the terminal cost if x(k+1) is a terminal state

path = '../../../results/cart_pole/ddpg/'  # path, where results are saved

algorithm = DDPG(env, t, dt, path=path) # instance of the DDPG algorithm

#algorithm.load() # can be used to load existing networks and data set

learning_steps = 5e5 # define training duration
algorithm.run_learning(learning_steps) # run reinforcment learning
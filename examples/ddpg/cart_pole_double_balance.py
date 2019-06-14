from pygent.environments import CartPoleDoubleSerial
from pygent.algorithms.ddpg import DDPG
import numpy as np

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 0.5*x1**2 + x2**2 + x3**2 + 0.01*x4**2 + 0.01*x5**2+ 0.01*x6**2 + 0.01*u1**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.01, 0.01), np.random.uniform(-0.1, 0.1),
          np.random.uniform(-0.1, 0.1), 0, 0, 0]
    return x0

t = 10 # time of an episode
dt = 0.01 # time step-size

env = CartPoleDoubleSerial(c_k, p_x0, dt, task='balance')

env.terminal_cost = 200 # define the terminal cost if x(k+1) is a terminal state

path = '../../../results/cart_pole_double_balance/ddpg/' # path, where results are saved

algorithm = DDPG(env, t, dt, path=path) # instance of the DDPG algorithm

#algorithm.load() # can be used to load existing networks and data set

learning_steps = 2e6 # define training duration
algorithm.run_learning(learning_steps) # run reinforcment learning

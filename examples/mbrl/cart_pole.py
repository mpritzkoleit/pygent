from pygent.environments import CartPole
from pygent.algorithms.ddpg import DDPG
from pygent.algorithms.mbrl import MBRL
from pygent.data import DataSet
from pygent.helpers import mapAngles
import numpy as np
import torch
import torch.nn as nn
# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = .5*(20.*x1**2 + 20*x2**2 + .01*x3**2 + .01*x4**2) + 0.01*u1**2
    return c

def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 100*x2**2 + 10*x3**2 + 10*x4**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.01, 0.01), np.random.uniform(0.99*np.pi, 1.01*np.pi), 0, 0]
    return x0


x0 = [0, np.pi, 0, 0]
t = 6 # time of an episode
dt = 0.02 # time step-size

env = CartPole(c_k, x0, dt)

path = '../../../results/mbrl/'  # path, where results are saved

rl_algorithm = MBRL(env, t, dt, path=path, warm_up=1000, horizon=2., use_mpc_plan=False, ilqr_print=True) # instance of the DDPG algorithm
#rl_algorithm.load()
rl_algorithm.run_learning(40)


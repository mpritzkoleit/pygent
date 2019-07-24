from pygent.environments import Pendulum
from pygent.algorithms.ddpg import DDPG
from pygent.algorithms.mbrl import MBRL
from pygent.data import DataSet
from pygent.helpers import mapAngles
import numpy as np
import torch
import torch.nn as nn
# define the incremental cost
def c_k(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + .1*x2**2 + 0.01*u1**2
    return c

def c_N(x):
    x1, x2 = x
    c = 100*x1**2 + 1*x2**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(0.99*np.pi, 1.01*np.pi), np.random.uniform(-.01, .01)]
    return x0


x0 = [np.pi, 0]
t = 6 # time of an episode
dt = 0.05 # time step-size

env = Pendulum(c_k, x0, dt)
env.uMax = env.uMax/3.5*5

path = '../../../results/mbrl/'  # path, where results are saved
rl_algorithm = MBRL(env, t, dt, path=path, horizon=4., fcost=c_N, warm_up=600, use_mpc_plan=False, use_feedback=True,
                    ilqr_print=True, ilqr_save=True) # instance of the DDPG algorithm
rl_algorithm.run_learning(50)


from pygent.environments import Acrobot
from pygent.algorithms.ddpg import DDPG
import numpy as np
import time

def cost(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = x1**2 + x2**2 + 0.01*x3**2 + 0.01*x4**2 + 0.05*u1**2
    return c


def x0fun():
    x0 = [np.random.uniform(0.99*np.pi, 1.01*np.pi), np.random.uniform(-.01, .01), 0, 0]
    return x0

t = 10.0
dt = 0.03

env = Acrobot(cost, x0fun, dt)
env.terminal_cost = 200

path = '../../../results/acrobot/ddpg/'
algorithm = DDPG(env, t, dt, path=path)
algorithm.load()
algorithm.run_learning(5e5)

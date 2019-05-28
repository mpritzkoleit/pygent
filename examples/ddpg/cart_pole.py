from pygent.environments import CartPole
from pygent.algorithms.ddpg import DDPG
import numpy as np


def cost(x_, u, x):
    x1, x2, x3, x4 = x
    u1, = u
    c = x1**2 + 2*x2**2 + 0.02*x3**2 + 0.01*x4**2 + 0.01*u1**2
    return c

def x0fun():
    x0 = [np.random.uniform(-0.001, 0.001), np.random.uniform(0.999*np.pi, 1.001*np.pi), 0, 0]
    return x0

t = 6
dt = 0.02

cartPole = CartPole(cost, x0fun, dt)
cartPole.terminal_cost = 200

path = '../../../results/ddpg/experiment1/'

algorithm = DDPG(cartPole, t, dt, path=path)
algorithm.run_learning(1e6)
from pygent.environments import CartPoleDoubleSerial
from pygent.algorithms.ddpg import DDPG
import numpy as np


def cost(x_, u, x):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 0.5*x1**2 + x2**2 + x3**2 + 0.01*x4**2 + 0.01*x5**2+ 0.01*x6**2 + 0.01*u1**2
    return c

def x0fun():
    x0 = [np.random.uniform(-0.01, 0.01), np.random.uniform(0.99*np.pi, 1.01*np.pi),
          np.random.uniform(0.99*np.pi, 1.01*np.pi), 0, 0, 0]
    return x0

t = 10
dt = 0.01

cartPole = CartPoleDoubleSerial(cost, x0fun, dt)
cartPole.terminal_cost = 200

path = '../../../results/cart_pole_double/ddpg/'

algorithm = DDPG(cartPole, t, dt, path=path)
algorithm.load()
algorithm.run_learning(2e6)

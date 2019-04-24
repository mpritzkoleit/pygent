from environments import CartPoleDoubleParallel
from algorithms.ddpg import DDPG
import numpy as np

def cost(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = (0.1*np.abs(x1) + 1 - np.cos(x2) + 1 - np.cos(x3) + 1e-1*x4**2 + 1e-1*x5**2 + 1e-1*x6**2 + 1e-3*u1**2)
    if np.abs(x1) > 1:
        c = 5
    return c


def x0fun():
    x0 = [np.random.uniform(-.05, .05), np.random.uniform(0.98 * np.pi, 1.02 * np.pi),
          np.random.uniform(0.98 * np.pi, 1.02 * np.pi), 0, 0, 0]
    return x0

cartPole = CartPoleDoubleParallel(cost, x0fun)
t = 15
dt = 0.03

path = '../../../results/ddpg/cart_pole_double_parallel/'

algorithm = DDPG(cartPole, t, dt, path=path)

#algorithm.run_episode()
algorithm.run_learning(10000)

from pygent.environments import Pendulum
from pygent.algorithms.mbrl import MBRL
import numpy as np

def c_k(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + 0.1*x2**2 + 0.01*u1**2
    return c

x0 = [np.pi, 0]

def x0fun():
    x0 = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-1, 1)]
    return x0

t = 6.
dt = 0.05

env = Pendulum(c_k, x0fun, dt)

path = '../../../results/mbrl/cart_pole/'

controller = MBRL(env, t, dt, path=path)
controller.run_learning(1000)

from environments import CartPole
from algorithms.mbrl import MBRL
import numpy as np

def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = .5 * (5*x1**2 + 2 * x2 ** 2 + 0.01 * x3 ** 2 + 0.01 * x4 ** 2 + 0.01 * u1 ** 2)
    return c

def c_N(x):
    x1, x2, x3, x4 = x
    c = .5*(50.*x1**2 + 20*x2**2 + 1*x3**2 + 1*x4**2)
    return c

x0 = [0, 0.1, 0, 0]

t = 5
dt = 0.01

cartPole = CartPole(c_k, x0, dt)

path = '../../../results/mbrl/cart_pole/'

controller = MBRL(cartPole, t, dt, path=path)
controller.run_learning(1000)

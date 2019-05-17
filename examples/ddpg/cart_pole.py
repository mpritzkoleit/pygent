from environments import CartPole
from algorithms.ddpg import DDPG
import numpy as np


def cost(x_, u, x):
    x1, x2, x3, x4 = x
    u1, = u
    c = 8.7*x1**2 + 8.7*x2**2 + 10e-5*x3**2 + 10e-5*x4**2 + 4.7*u1**2
    return c

def x0fun():
    x0 = [np.random.uniform(-0.1, 0.1), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
    return x0

t = 10
dt = 0.02

cartPole = CartPole(cost, x0fun, dt)

path = '../../../results/ddpg/experiment55/'

algorithm = DDPG(cartPole, t, dt, path=path, warm_up=100, plotInterval=1)
algorithm.load()
algorithm.run_learning(2000)

x0 = [0.0, np.pi, 0, 0]
algorithm.run_controller(x0)
algorithm.plot()
algorithm.animation()

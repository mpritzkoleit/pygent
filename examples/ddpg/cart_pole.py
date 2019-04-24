from environments import CartPole
from algorithms.ddpg import DDPG
import numpy as np


def cost(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = (x1 - np.sin(x2))**2 + (1 - np.cos(x2))**2 + 0.1*x3**2

    if abs(x1)>1:
        c+=100
    return c

x0 = [0.0, np.pi, 0, 0]
def x0fun():
    x0 = [np.random.uniform(-0.05, 0.05), np.random.uniform(0.99*np.pi, 1.01*np.pi), 0, 0]
    return x0

cartPole = CartPole(cost, x0fun)
t = 6
dt = 0.01

path = '../../../results/ddpg/cart_pole/experiment1/'

algorithm = DDPG(cartPole, t, dt, path=path, a_lr=1e-3, tau=0.005, batch_size=128)
algorithm.load()
algorithm.run_learning(4000)
algorithm.plot()
algorithm.animation()

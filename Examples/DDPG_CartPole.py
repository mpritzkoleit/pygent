from Environments import CartPole
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt


def cost(x_, u_, x):
    x1, x2, x3, x4 = x_
    u1, = u_
    sigma_c = 0.25
    c = np.linalg.norm(np.matrix([[x1 - np.sin(x2)], [1 - np.cos(x2)]]),2)
    c = 1 - np.exp(-1/(2*sigma_c**2)*c)
    if abs(x1)>1.0:
        c = 100
    return c

x0 = [0, np.pi/2, 0, 0]
def x0fun():
    x0 = [np.random.uniform(-0.5, 0.5), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
    return x0

cartPole = CartPole(cost, x0fun)
t = 10
dt = 0.02

path = '../Results/DDPG/cartPole/Experiment28/'

algorithm = DDPG(cartPole, t, dt, path=path)
algorithm.load()
algorithm.run_learning(3000)

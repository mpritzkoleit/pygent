from Environments import CartPole
from DDPG import DDPG
import numpy as np
import matplotlib.pyplot as plt


def cost(x_, u_, x):
    x1, x2, x3, x4 = x_
    u1, = u_
    #sigma_c = 1
    c = (x1 - np.sin(x2))**2 + (1 - np.cos(x2))**2 + 0.1*x3**2
    #c = 1 - np.exp(-1/(2*sigma_c**2)*c)
    #c = -0.5*np.cos(x2) + 0.03*u1**2 + 0.015*np.abs(x1) + 0.2*x4**2
    if abs(x1)>1:
        c+=100
    return c

x0 = [0, np.pi/2, 0, 0]
def x0fun():
    x0 = [np.random.uniform(-0.25, 0.25), np.random.uniform(0.95*np.pi, 1.05*np.pi), 0, 0]
    return x0

cartPole = CartPole(cost, x0fun)
t = 6
dt = 0.03

path = '../Results/DDPG/cartPole/Experiment41/'

algorithm = DDPG(cartPole, t, dt, path=path, costScale=33)#, a_lr=1e-3, tau=0.005, batch_size=128)
algorithm.load()
algorithm.run_learning(2000)

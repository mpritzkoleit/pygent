from pygent.environments import CartPole, Environment
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = 0.5*x1**2 + x2**2 + 0.02*x3**2 + 0.05*x4**2 + 0.05*u1**2
    return c

def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 100*x2**2 + 10*x3**2 + 10*x4**2
    return c


x0 = [0, np.pi, 0, 0]


t = 10
dt = 0.02
cartPole = CartPole(c_k, x0, dt)

path = '../../../results/ilqr/cart_pole/'

controller = iLQR(cartPole, t, dt, path=path, fcost = c_N, constrained=True)
controller.run_optim()
controller.run(x0)
controller.plot()
plt.show()
controller.animation()

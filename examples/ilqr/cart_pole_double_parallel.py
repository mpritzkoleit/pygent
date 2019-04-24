from environments import CartPoleDoubleParallel
from algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def cost(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 0.5*(2*x1**2 + 10.*x2**2 + 10.*x3**2 + 0.5*x4**2 + .1*u1**2)
    return c


def finalcost(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 0.5*(100.*x1**2 + 100.*x2**2 + 100.*x3**2)
    return c


x0 = [0, np.pi, np.pi, 0, 0, 0]

cartPole = CartPoleDoubleParallel(cost, x0)
t = 8
dt = 0.01

path = '../../../results/ilqr/cart_pole_doube_parallel/'
controller = iLQR(cartPole, t, dt, constrained=True, fcost=finalcost, path=path)
controller.run_optim()

controller.plot()
plt.show()
controller.animation()

from environments import CartPole
from algorithms.ilqr import NMPC2
import numpy as np
import matplotlib.pyplot as plt

def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = .5*(10.*x1**2 + 20*x2**2 + .01*x3**2 + .01*x4**2) + 0.01*u1**2
    return c

x0 = [0, np.pi, 0, 0]

cartPole = CartPole(c_k, x0)
t = 6
dt = 0.01

path = '../../../results/nmpc/cart_pole/'

controller = NMPC2(cartPole, t, dt, horizon=2, path=path, constrained=True, maxIters=10, fastForward=True)
controller.run_mpc()
controller.plot()
plt.show()
controller.animation()

from pygent.environments import Pendulum
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def c_k(x, u, mod):
    x1, x2 = x
    u1, = u
    c = x1**2 + 1e-2*x2**2 + 1e-2*u1**2
    return c

def c_N(x):
    x1, x2 = x
    c = 500.*x1**2 + 10.*x2**2
    return c

x0 = [np.pi, 0]

t = 10
dt = 0.05

pendulum = Pendulum(c_k, x0, dt)

path = '../../../results/pendulum/ilqr/'

controller = iLQR(pendulum, t, dt, constrained=True, path=path)
controller.run_disk(x0)
controller.run_optim()
controller.plot()
plt.show()
controller.animation()

from environments import Pendulum
from algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def c_k(x, u, mod):
    x1, x2 = x
    u1, = u
    c = x1**2 + 1e-1*x2**2 + 1e-3*u1**2
    return c

def c_N(x):
    x1, x2 = x
    c = 500.*x1**2 + 10.*x2**2
    return c

x0 = [np.pi, 0]

pendulum = Pendulum(c_k, x0)

t = 10
dt = 0.01

path = '../../../results/ilqr/pendulum/'

controller = iLQR(pendulum, t, dt, constrained=True, fcost=c_N, path=path)
#controller.run(x0)
controller.run_optim()
controller.plot()
plt.show()
controller.animation()

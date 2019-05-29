from pygent.environments import Pendulum
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def c_k(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + 0.01*x2**2 + 0.01*u1**2
    return c

x0 = [np.pi, 0]

t = 10
dt = 0.05

env = Pendulum(c_k, x0, dt)

path = '../../../results/pendulum/ilqr/'

controller = iLQR(env, t, dt, constrained=True, path=path)
controller.run_optim()
controller.plot()
plt.show()
controller.animation()

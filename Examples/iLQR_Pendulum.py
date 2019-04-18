from Environments import Pendulum
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt

def cost(x_, u_, x):
    x1, x2 = x_
    u1, = u_
    c = 0.5*(10*x1**2 + .001*x2**2 + 1*u1**2)
    return c

def finalcost(x):
    x1, x2 = x
    c = 0.5*(10*x1**2 + 1.*x2**2)
    return c

x0 = [np.pi, 0]

pendulum = Pendulum(cost, x0)

t = 10
dt = 0.01

path = '../Results/iLQR/Pendulum/'

controller = iLQR(pendulum, t, dt, constrained=True, path=path)
controller.run_optim()

controller.plot()
plt.show()
controller.animation()

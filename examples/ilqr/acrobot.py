from environments import Acrobot
from algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def cost(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = .5*(10*x1**2 + 2*x2**2 + 0.01*x3**2 + 0.01*x4**2 + 1*u1**2)
    return c

def finalcost(x):
    x1, x2, x3, x4 = x
    c = .5*(10*x1**2 + 2*x2**2 + 0.01*x3**2 + .01*x4**2)
    return c


x0 = [np.pi, 0., 0., 0.]

acrobot = Acrobot(cost, x0)
t = 6
dt = 0.03

path = '../../../results/ilqr/acrobot/'

controller = iLQR(acrobot, t, dt, fcost=finalcost, path=path, constrained=True)
controller.run_optim()
#controller.run(x0)
controller.plot()
plt.show()
controller.animation()

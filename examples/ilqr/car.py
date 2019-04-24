from environments import Car
from algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def cost(x, u):
    x1, x2, x3, x4 = x
    u1, u2 = u
    c = 1e-3*x1**2 + 1e-3*x2**2 + 1e-2*u1**2 + 1e-4*u2**2
    return c/0.03

def finalcost(x):
    x1, x2, x3, x4 = x
    c = .1*x1**2 + .1*x2**2 + 1.*x3**2 + .3*x4**2
    return c/0.03

x0 = [1., 1., 3./2.*np.pi, 0.]

car = Car(cost, x0)

t = 15
dt = 0.03

controller = iLQR(car, t, dt, fcost=finalcost, fastForward=True, constrained=True)
#controller.run_optim()
controller.run(x0)

controller.plot()
plt.show()
#controller.animation()

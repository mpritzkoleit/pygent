from Environments import Pendulum
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt

def cost(x_, u_, x):
    x1, x2 = x_
    u1, = u_
    c = 0.5*(1.*x1**2 + 0.01*x2**2 + .1*u1**2)
    return c

x0 = [np.pi, 0]

cartPole = Pendulum(cost, x0)
t = 3
dt = 0.01

controller = iLQR(cartPole, t, dt)
#controller.run_optim()
controller.run(x0)

controller.plot()
plt.show()
controller.animation()

from Environments import CartPole
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt
def cost(x_, u_, x):
    x1, x2, x3, x4 = x_
    u1, = u_
    c = 0.5*(0.5*x1**2 + 2.*x2**2 + 0.01*x3**2 + 0.01*x4**2 + .05*u1**2)
    return c


x0 = [0, np.pi, 0, 0]

cartPole = CartPole(cost, x0)
t = 6
dt = 0.01

controller = iLQR(cartPole, t, dt)
controller.run_optim()
#controller.run(x0)
controller.plot()
plt.show()
controller.animation()

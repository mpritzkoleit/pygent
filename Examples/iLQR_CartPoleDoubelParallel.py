from Environments import CartPoleDoubleParallel
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt
def cost(x_, u_, x):
    x1, x2, x3, x4, x5, x6 = x_
    u1, = u_
    c = 0.5*(10.*x1**2 + 10.*x2**2 + 10.*x3**2 + 0.01*x4**2 + 0.01*x5**2 + 0.01*x6**2 + .03*u1**2)
    return c

x0 = [0, np.pi, np.pi, 0, 0, 0]

cartPole = CartPoleDoubleParallel(cost, x0)
t = 5
dt = 0.01

controller = iLQR(cartPole, t, dt, maxIters=200, fastForward=True)
controller.run_optim()

controller.plot()
plt.show()
controller.animation()

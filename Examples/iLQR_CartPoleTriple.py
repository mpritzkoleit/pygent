from Environments import CartPoleTriple
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt
def cost(x_, u_, x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x_
    u1, = u_
    c = 0.5*(1.*x1**2 + 2.*x2**2 + 2.*x3**2 + 2.*x4**2 + 0.01*x5**2 + 0.01*x6**2 + 0.01*x7**2 + 0.01*x8**2 + .01*u1**2)
    return c

def finalcost(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    c = 0.5*(1.*x1**2 + 10.*x2**2 + 10.*x3**2 + 10.*x4**2 + 1.*x5**2 + 1.*x6**2 + 1.*x7**2 + 1.*x8**2)
    return c

x0 = [0, np.pi, np.pi, np.pi, 0, 0, 0, 0]

cartPole = CartPoleTriple(cost, x0)
t = 6
dt = 0.01

path = '../Results/iLQR/CartPoleTriple2/'
controller = iLQR(cartPole, t, dt, constrained=False, fcost=finalcost, path=path)
controller.run_optim()
#controller.run(x0)
controller.plot()
plt.show()
controller.animation()

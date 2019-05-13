from environments import CartPoleDoubleSerial
from algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

def cost(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 15*x1**2 + 10*x2**2 + 10*x3**2 + .01*u1**2
    return c


def finalcost(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2
    return c


x0 = [0, np.pi, np.pi, 0, 0, 0]

t = 3.5
dt = 0.005

cartPole = CartPoleDoubleSerial(cost, x0, dt)

path = '../../../results/ilqr/cart_pole_double_serial/'
controller = iLQR(cartPole, t, dt, constrained=True, fcost=finalcost, path=path)
controller.run(x0)
controller.run_optim()
controller.plot()
plt.show()
controller.animation()

from Environments import CartPole
from iLQR import NMPC
import numpy as np
import matplotlib.pyplot as plt

def cost(x_, u_, x):
    x1, x2, x3, x4 = x_
    u1, = u_
    c = .5*(1*x1**2 + 5*x2**2 + 0.01*x3**2 + 0.01*x4**2 + 0.5*u1**2)
    return c


x0 = [0, np.pi, 0, 0]

cartPole = CartPole(cost, x0)
t = 3
dt = 0.001

path = '../Results/NMPC/CartPole/'

controller = NMPC(cartPole, t, dt, horizon=3000, path=path, constrained=False)
controller.run_mpc()
controller.plot()
plt.show()
controller.animation()

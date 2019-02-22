from Environments import CartPoleDoubleSerial
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt
def cost(x_, u_, x):
    x1, x2, x3, x4, x5, x6 = x_
    u1, = u_
    c = 0.5*(10*x1**2 + 10.*x2**2 + 10.*x3**2 + 0.01*x4**2 + 0.01*x5**2 + 0.01*x6**2 + .1*u1**2)
    return c

def finalcost(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 0.5*(10.*x1**2 + 10.*x2**2 + 10.*x3**2 + 1.*x4**2 + 1.*x5**2 + 1.*x6**2)
    return c

x0 = [0, np.pi, np.pi, 0, 0, 0]

cartPole = CartPoleDoubleSerial(cost, x0)
t = 3.5
dt = 0.01

path = '../Results/iLQR/CartPoleDoubleSerial2/'
controller = iLQR(cartPole, t, dt, constrained=True, fcost=finalcost, path=path)
controller.run_optim()
#controller.run(x0)
controller.plot()
plt.show()
controller.animation()

from Environments import CartPoleDoubleSerial
from iLQR import iLQR
import numpy as np
import matplotlib.pyplot as plt
def cost(x_, u_, x):
    x1, x2, x3, x4, x5, x6 = x_
    u1, = u_
    c = 5*x1**2 + 10*x2**2 + 10*x3**2 + .1*u1**2
    #xtip = x1 - np.sin(x2) + np.sin(x3)
    #ytip = np.cos(x2) + np.cos(x3)
    #c = (xtip)**2 + (ytip-2)**2 + 0.01*u1**2
    return c

def finalcost(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2
    return c

x0 = [0, np.pi, np.pi, 0, 0, 0]

cartPole = CartPoleDoubleSerial(cost, x0)
t = 3.5
dt = 0.01

path = '../Results/iLQR/CartPoleDoubleSerial/'
controller = iLQR(cartPole, t, dt, constrained=True, fcost=finalcost, path=path)#, maxIters=200)
controller.run_optim()
#controller.run(x0)
controller.plot()
plt.show()
controller.animation()

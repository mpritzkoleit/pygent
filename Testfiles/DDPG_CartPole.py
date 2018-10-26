from Environments import CartPole
from DDPG import DDPG
import numpy as np

def cost(x_, u, x):
    x1, x2, x3, x4 = x
    c = 8.7*x1**2 + 8.7*x2**2 + 10e-5*x3**2 + 10e-5*x4**2 + 4.7*u[0]**2
    if abs(x1) > 1 or abs(x4) > 25:
        terminate = True
    else:
        terminate = False
    return c, terminate


x0 = [np.random.uniform(-.5, .5), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
def x0fun():
    x0 = [np.random.uniform(-0.5, 0.5), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
    return x0

cartPole = CartPole(cost, x0fun)
t = 6.0
dt = 0.03


xDim = 5
uDim = 1
uMax = 2

algorithm = DDPG(cartPole, xDim, uDim, uMax, t, dt, 20)

#algorithm.run_episode()
algorithm.run_learning(10000)

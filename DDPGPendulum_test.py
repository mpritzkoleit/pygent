from Environments import Pendulum
from DDPG import DDPG
import numpy as np

def cost(x_, u, x):
    x1, x2 = x
    c = x1**2 + 1e-2*x2**2 + 1e-3*u[0]**2
    if abs(x2) > 10:
        terminate = True
    else:
        terminate = False
    return -c, terminate


x0 = [np.random.uniform(0.99 * np.pi, 1.01 * np.pi), 0]

def x0fun():
    x0 = [np.random.uniform(0.95 * np.pi, 1.05 * np.pi), np.random.uniform(-.05, .05)]
    return x0

cartPole = Pendulum(cost, x0fun)

t = 6
dt = 0.05


xDim = 2
uDim = 1
uMax = 1

algorithm = DDPG(cartPole, xDim, uDim, uMax, t, dt, 100)

#algorithm.run_episode()
algorithm.run_learning(5000)

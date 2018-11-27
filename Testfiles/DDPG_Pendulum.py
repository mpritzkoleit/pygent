from Environments import Pendulum
from DDPG import DDPG
import numpy as np

def cost(x_, u, x):
    x1, x2 = x
    c = (x1**2 + 1e-1*x2**2 + 1e-3*u[0]**2)
    terminate = False
    if abs(x2) > 8:
        terminate = True
    return c, terminate


def x0fun():
    x0 = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-1.0, 1.0)]
    return x0


pendulum = Pendulum(cost, x0fun)

t = 10.0
dt = 0.05


xDim = 3
uDim = 1
uMax = 1

algorithm = DDPG(pendulum, xDim, uDim, uMax, t, dt, 25)

#algorithm.run_episode()
algorithm.run_learning(1000)

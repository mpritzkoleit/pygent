from Environments import Pendulum
from NFQ import NFQ
import numpy as np
import matplotlib.pyplot as plt
def cost(x_, u, x):
    x1, x2 = x
    if abs(x2) > 10:
        c = 1.
    elif abs(x1) < 0.1  and abs(x2) < 0.5:
        c = 0.
    else:
        c = 0.01
    return c

x0 = [np.pi, 0]

pendulum = Pendulum(cost, x0)
t = 6
dt = 0.05
controls = np.array([-5, 0, 5]).T
xGoal = [0, 0]
gamma = 0.99
eps = 0.1
netStructure = [3, 20, 20, 1]


algorithm = NFQ(pendulum, controls, xGoal, t, dt, netStructure, eps, gamma)

algorithm.run_episode()
algorithm.run_learning(200)
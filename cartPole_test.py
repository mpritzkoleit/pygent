from Environments import CartPole
from NFQ import NFQ
import numpy as np

def cost(x_, u, x):
    x1, x2, x3, x4 = x
    if abs(x1) > 0.25:
        c = 1
    elif abs(x2) < 0.2 and abs(x1) < 0.1:
        c = 0
    else:
        c = 0.01
    return c

x0 = [0, np.pi, 0, 0]

cartPole = CartPole(cost, x0)

t = 10
dt = 0.05
controls = np.array([-1, 0, 1]).T
xGoal = [0, 0, 0, 0]
gamma = 0.99
eps = 0.0
netStructure = [5, 20, 20, 1]


algorithm = NFQ(cartPole, controls, xGoal, t, dt, netStructure, eps, gamma)

algorithm.run_episode()
algorithm.run_learning(400)

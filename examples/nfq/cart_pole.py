from environments import CartPole
from algorithms.nfq import NFQ
import numpy as np

def cost(x):
    x1, x2, x3, x4 = x
    if abs(x1) > 0.25:
        c = 1
    elif abs(x2) < 0.2 and abs(x1) < 0.1:
        c = 0
    else:
        c = 0.01
    return c, False

x0 = [0, np.pi, 0, 0]

cartPole = CartPole(cost, x0)

t = 10
dt = 0.05
controls = np.array([-1, 0, 1]).T
xGoal = [0, 0, 0, 0]
gamma = 0.99
eps = 0.0
h_layers = [20, 20] # hidden layer dimensions in Q-Network


algorithm = NFQ(cartPole, controls, xGoal, t, dt, h_layers, eps, gamma)

algorithm.run_episode()
algorithm.run_learning(400)

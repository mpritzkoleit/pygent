from Environments import CartPole
from DDPG_HER import DDPG_HER
import numpy as np
import matplotlib.pyplot as plt

def cost(x_, u_, x):
    x1, x2, x3, x4 = x_
    u1, = u_
    c = np.linalg.norm(np.matrix([[x1 - np.sin(x2)], [1 - np.cos(x2)]]),2) + 0.01*x3**2 + 0.01*x4 + 0.01*u1
    #c = .5 * (0.5*x1**2 + 2*x2**2 + 0.01*x3**2 + 0.01*x4**2 + .1*u1** 2)
    if abs(x1)>1.0:
        c = 100
    return c


x0 = [0, np.pi/2, 0, 0]
def x0fun():
    x0 = [np.random.uniform(-0.5, 0.5), np.random.uniform(0.9*np.pi, 1.1*np.pi), 0, 0]
    return x0

cartPole = CartPole(cost, x0fun)
t = 10
dt = 0.03

xGoal = [0., 0., 0., 0.]

path = '../Results/DDPG_HER/cartPole/Experiment1/'

algorithm = DDPG_HER(cartPole, t, dt, xGoal, path=path)
algorithm.load()
algorithm.run_learning(1000)
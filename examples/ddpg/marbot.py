from pygent.environments import MarBot
from pygent.algorithms.ddpg import DDPG
import numpy as np


def cost(x, u, x_):
    x1, x2, x3, x4 = x_
    u1, = u
    c = 8.7*x1**2 + 8.7*x2**2 + 10e-5*x3**2 + 10e-5*x4**2 + 4.7*u1**2
    return c

def x0fun():
    x0 = [np.random.uniform(-1, 1), np.random.uniform(-1., 1.), 0, 0]
    return x0

t = 10
dt = 0.02

robot = MarBot(cost, x0fun, dt)

path = '../../../results/ddpg/marbot/experiment1/'

algorithm = DDPG(robot, t, dt, path=path, warm_up=10000)
algorithm.load()
algorithm.run_learning(2000)


x0 = [0., 0., 0.3, 0.]
algorithm.run_controller(x0)
algorithm.plot()
algorithm.animation()
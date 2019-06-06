from pygent.environments import Pendulum
from pygent.algorithms.ddpg import DDPG
import numpy as np
import time
def cost(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + 0.1*x2**2 + 0.05*u1**2
    return c


def x0fun():
    x0 = [np.random.uniform(0.999*np.pi, 1.001*np.pi), np.random.uniform(-0.001,0.001)]
    return x0

t = 10.0
dt = 0.05

pendulum = Pendulum(cost, x0fun, dt)

path = '../../../results/pendulum/ddpg/'
algorithm = DDPG(pendulum, t, dt, path=path)
start = time.time()
algorithm.load()
algorithm.run_learning(1e5)
end = time.time()
print('Training duration: %.2f s' % (end - start))
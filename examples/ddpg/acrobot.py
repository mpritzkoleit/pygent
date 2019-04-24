from environments import Acrobot
from algorithms.ddpg import DDPG
import numpy as np
import time

def cost(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    xtip = -np.sin(x1) - np.sin(x1 + x2)
    ytip = np.cos(x1) + np.cos(x1 + x2)
    ytip_ref = 2
    xtip_ref = 0
    c = (xtip-xtip_ref)**2 + (ytip-ytip_ref)**2
    c += 0.1*x3**2 + 0.1*x4**2 + 0.01*u1**2 + x1**2 + x2**2
    return c


def x0fun():
    x0 = [np.random.uniform(0.9*np.pi, 1.1*np.pi), np.random.uniform(-.1, .1), 0, 0]
    return x0


pendulum = Acrobot(cost, x0fun)

t = 10.0
dt = 0.03

path = '../../../results/ddpg/acrobot/experiment1/'
algorithm = DDPG(pendulum, t, dt, path=path)
start = time.time()
algorithm.load()
algorithm.run_learning(1500)
#algorithm.run_controller(x0fun())
end = time.time()
print('Training duration: %.2f s' % (end - start))
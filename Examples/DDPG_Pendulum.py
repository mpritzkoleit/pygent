from Environments import Pendulum
from DDPG import DDPG
import numpy as np
import time
def cost(x_, u_, x):
    x1, x2 = x_
    u1, = u_
    c = (x1**2 + 1e-1*x2**2 + 1e-3*u1**2)
    return c


def x0fun():
    x0 = [np.random.uniform(-np.pi, np.pi), np.random.uniform(-1.0, 1.0)]
    return x0


pendulum = Pendulum(cost, x0fun)

t = 10.0
dt = 0.05

path = '../Results/DDPG/Pendulum/'
algorithm = DDPG(pendulum, t, dt, path=path)
start = time.time()
algorithm.run_learning(2000)
#algorithm.run_controller(x0fun())
end = time.time()
print('Training duration: %.2f s' % (end - start))
from pygent.environments import CartPole
from pygent.algorithms.nmpc import NMPC
import numpy as np
import matplotlib.pyplot as plt

# incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = .5*(20.*x1**2 + 20*x2**2 + .01*x3**2 + .01*x4**2) + 0.01*u1**2
    return c

x0 = [0, np.pi, 0, 0]

t = 6.
horizon = 2
dt = 0.02

env = CartPole(c_k, x0, dt)
mpc_env = CartPole(c_k, x0, dt)

path = '../../../results/nmpc/cart_pole/'

algorithm = NMPC(env, mpc_env, t, dt, horizon, path=path, ilqr_print=False, ilqr_save=False, step_iterations=1)
# reset environment/agent to initial state, delete history
algorithm.run()

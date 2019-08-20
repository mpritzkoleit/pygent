from pygent.environments import CartPoleTriple
from pygent.algorithms.nmpc import NMPC
import numpy as np
import matplotlib.pyplot as plt

# incremental cost
def c_k(x, u):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    u1, = u
    c = .5*(20.*x1**2 + 20*(x2-np.pi)**2 + .01*x5**2 + .01*x6**2) + 0.01*u1**2
    return c

x0 = [0.5, np.pi, np.pi, np.pi, 0, 0, 0, 0]

t = 6.
horizon = 2
dt = 0.01

env = CartPoleTriple(c_k, x0, dt)
mpc_env = CartPoleTriple(c_k, x0, dt)

path = '../../../results/nmpc/cart_pole_double/'

algorithm = NMPC(env, mpc_env, t, dt, horizon, path=path, ilqr_print=False, ilqr_save=False, step_iterations=1)
# reset environment/agent to initial state, delete history
algorithm.run()

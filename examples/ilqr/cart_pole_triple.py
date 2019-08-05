from pygent.environments import CartPoleTriple
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    u1, = u
    #c = 15*x1**2 + 10*x2**2 + 10*(x3-np.pi)**2 + .1*u1**2
    c = 15 * x1 ** 2 + 10 * x2 ** 2  + .1 * u1 ** 2
    return c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    c = 100*x1**2 + 100*x2**2 + 100*(x3-np.pi)**2 + 100*x4**2 + 10*x5**2 + 10*x6**2 + 10*x7**2 + 10*x8**2
    return c

# initial state value
x0 = [0, 0, 0, 0, 0, 0, 0, 0]

t = 3.5 # simulation time
dt = 0.005 # time step-size

env = CartPoleTriple(c_k, x0, dt)

path = '../../../results/ilqr/cart_pole_triple/'  # path, where results are saved

algorithm = iLQR(env, t, dt, constrained=True, fcost=c_N, path=path, finite_diff=True) # instance of the iLQR algorithm

algorithm.run_optim()  # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()
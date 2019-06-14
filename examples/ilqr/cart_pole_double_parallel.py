from pygent.environments import CartPoleDoubleParallel
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 0.5*(2*x1**2 + 10.*x2**2 + 10.*x3**2 + 0.5*x4**2 + .1*u1**2)
    return c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 0.5*(100.*x1**2 + 100.*x2**2 + 100.*x3**2)
    return c

# initial state value
x0 = [0, np.pi, np.pi, 0, 0, 0]

t = 8 # simulation time
dt = 0.01 # time step-size

env = CartPoleDoubleParallel(c_k, x0, dt)

path = '../../../results/ilqr/cart_pole_double_parallel/'  # path, where results are saved

algorithm = iLQR(env, t, dt, constrained=True, fcost=c_N, path=path)

algorithm.run_optim() # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()
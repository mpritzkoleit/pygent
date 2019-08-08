from pygent.environments import CartPoleDoubleSerial
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    #c = 15*x1**2 + 10*x2**2 + 10*x3**2 + .2*u1**2
    c = 500*x1**2 + 900*x2**2 + 900*x3**2 + 20*x4**2 + 100*x5**2 + 100*x6**2 + 10*u1**2
    return c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2 + 10*x4**2 + 10*x5**2 + 10*x6**2
    return c

# initial state value
x0 = [0, np.pi, np.pi, 0, 0, 0]

t = 5 # simulation time
dt = 0.01 # time step-size

env = CartPoleDoubleSerial(c_k, x0, dt)

path = '../../../results/ilqr/cart_pole_double_serial/'

algorithm = iLQR(env, t, dt, constrained=True, fcost=None, path=path, maxIters=1000) # instance of the iLQR algorithm
#algorithm.run_disk(x0)
algorithm.run_optim() # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()
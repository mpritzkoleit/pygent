from pygent.environments import CartPole, Environment
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u, t, mod):
    x1, x2, x3, x4 = x
    u1, = u
    c = 10*x1**2 + 5*x2**2  + 0.01*x3**2 + 0.01*x4**2 + 0.1*u1**2
    return 0.5*c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2 + 100*x4**2
    return 0.5*c

# initial state value
x0 = [0, np.pi, 0, 0]

t = 5 # simulation time
dt = 0.02 # time step-size
env = CartPole(c_k, x0, dt)
path = './results/ilqr/cart_pole/' # path, where results are saved
algorithm = iLQR(env, t, dt, path=path, fcost=c_N, constrained=True) # instance of the iLQR algorithm
algorithm.run_optim() # run trajectory optimization
# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()
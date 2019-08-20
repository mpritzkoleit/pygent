from pygent.environments import Acrobot
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = 5*x1**2 + x2**2 + 0.01*x3**2 + 0.01*x4**2 + 0.005*u1**2
    return c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 100*x2**2 + 10*x3**2 + 10*x4**2
    return c

# initial state value
x0 = [np.pi, 0., 0., 0.]

t = 6 # simulation time
dt = 0.03 # time step-size

env = Acrobot(c_k, x0, dt)
env.uMax = env.uMax*0.7

path = '../../../results/ilqr/acrobot2/' # path, where results are saved

algorithm = iLQR(env, t, dt, fcost=c_N, path=path, constrained=True) # instance of the iLQR algorithm

algorithm.run_optim() # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()
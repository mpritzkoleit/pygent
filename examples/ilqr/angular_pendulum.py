from pygent.environments import AngularPendulum
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2, x3 = x
    u1, = u
    c = (1-x1) + 0.1*x3**2 + 0.01*u1**2
    return c

def c_N(x):
    x1, x2, x3 = x
    c = 100*(1-x1) +  0.1*x3**2
    return c

# initial state value
x0 = [-1, 0, 0]

t = 5 # simulation time
dt = 0.05 # time step-size

env = AngularPendulum(c_k, x0, dt)

path = '../../../results/pendulum/ilqr/' # path, where results are saved

algorithm = iLQR(env, t, dt, fcost=c_N, constrained=True, path=path) # instance of the iLQR algorithm

algorithm.run_optim() # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()

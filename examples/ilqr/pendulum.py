from pygent.environments import Pendulum
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + 0.1*x2**2 + 0.05*u1**2
    return c

# initial state value
x0 = [np.pi, 0]

t = 10 # simulation time
dt = 0.05 # time step-size

env = Pendulum(c_k, x0, dt)

path = '../../../results/pendulum/ilqr/' # path, where results are saved

algorithm = iLQR(env, t, dt, constrained=True, path=path) # instance of the iLQR algorithm

algorithm.run_optim() # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()

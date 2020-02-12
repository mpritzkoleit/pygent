from pygent.environments import Pendulum
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u, t, mod):
    x1, x2 = x
    u1, = u
    c = x1**2 + .01*x2**2 + 0.5*(1000*mod.exp(-t*15)+ 1)*u1**2
    return c

def c_N(x):
    x1, x2 = x
    c = 100*x1**2 + 1*x2**2
    return c

# initial state value
x0 = [np.pi, 0]

t = 6 # simulation time
dt = 0.05 # time step-size

env = Pendulum(c_k, x0, dt)

env.uMax = env.uMax/3.5*5

path = './results/pendulum/ilqr/' # path, where results are saved

algorithm = iLQR(env, t, dt, fcost=c_N, constrained=True, path=path, finite_diff=True) # instance of the iLQR algorithm

algorithm.run_optim() # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
algorithm.animation()

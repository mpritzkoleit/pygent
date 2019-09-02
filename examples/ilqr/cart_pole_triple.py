from pygent.environments import CartPoleTriple
from pygent.algorithms.ilqr import iLQR
import numpy as np
import matplotlib.pyplot as plt

# define the incremental cost
def c_k(x, u, mod):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    u1, = u
    c = 2*x1**2 + 1*x2**2 + 0.5*x3**2 + 0.5*x4**2
    c += 0.02*x5**2 + 0.05*x6**2 + 0.05*x7**2 + 0.05*x8**2 + 0.05*u1**2
    return c

# define the final cost at step N
def c_N(x, mod):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    c = 150*(x1-0)**2 + 100*(x2-0*mod.pi)**2 + 100*(x3-0*mod.pi)**2 + 100*(x4-0*mod.pi)**2 + 10*x5**2 + 10*x6**2 + 10*x7**2 + 10*x8**2
    return 10*c

# initial state value
x0 = [0., np.pi, np.pi, np.pi, 0, 0, 0, 0]

t = 3.5 # simulation time
dt = 0.002 # time step-size

env = CartPoleTriple(c_k, x0, dt)
env.uMax = 25.
path = '../../../results/ilqr/swingup_0_002/'  # path, where results are saved

algorithm = iLQR(env, t, dt, constrained=True, fcost=c_N, path=path, maxIters=100, fastForward=False) # instance of the iLQR algorithm#
algorithm.run_disk(x0)
algorithm.run_optim()  # run trajectory optimization

# plot trajectories
algorithm.plot()
plt.show()
# create an animation
#system
#algorithm.animation()
from pygent.environments import CartPole
from pygent.algorithms.mbrl import MPCAgent
import numpy as np
import matplotlib.pyplot as plt

def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = .5*(10.*x1**2 + 20*x2**2 + .01*x3**2 + .01*x4**2) + 0.01*u1**2
    return c

x0 = [0, np.pi, 0, 0]

t = 6
dt = 0.05

env = CartPole(c_k, x0, dt)
 

path = '../../../results/nmpc/cart_pole/'

agent = MPCAgent(1, env, 2., dt, path=path)
# reset environment/agent to initial state, delete history
env.reset(x0)
agent.reset()
cost = []
for _ in range(int(t/dt)):
    # agent computes control/action
    u = agent.take_action(dt, env.x)

    # simulation of environment
    c = env.step(u, dt)
    cost.append(c)
env.plot()
plt.show()

env.animation()

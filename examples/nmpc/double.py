from pygent.environments import CartPoleDoubleSerial
from pygent.algorithms.mbrl import MPCAgent
import numpy as np
import matplotlib.pyplot as plt

def cost(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 5*x1**2 + 10*x2**2 + 10*x3**2 + .01*u1**2
    return c


def finalcost(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2
    return c


x0 = [0, np.pi, np.pi, 0, 0, 0]

t = 3.5
dt = 0.005

env = CartPoleDoubleSerial(cost, x0, dt)

path = '../../../results/nmpc/cart_pole_double_serial/'

agent = MPCAgent(1, env, 1.5, dt, path=path, init_iterations=20)
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

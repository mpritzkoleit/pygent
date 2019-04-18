import numpy as np
import matplotlib.pyplot as plt
import torch
from Environments import StateSpaceModel
from Agents import FeedBack

def ode(t, x, u):
    dx1dt =  x[1]
    dx2dt =  x[2]
    dx3dt =  - 8*(u[0] + u[1])
    return np.array([dx1dt, dx2dt, dx3dt])

def cost(x_, u_, x):
    x = torch.Tensor(x_)
    u = torch.Tensor(u_)
    c = x.dot(x) + u.dot(u)
    return c, False

def mu(x):
    u1 = x[0]
    u2 = x[1] + x[2]
    return np.array([u1, u2])

t0 = 0
tf = 20
dt = 0.05
tt = np.arange(t0, tf, dt)

x0 = torch.Tensor([2, 1, -2.5])
env = StateSpaceModel(ode, cost, x0)
m = 2
agent = FeedBack(mu, m)

for t in tt:
    u = agent.take_action(dt, env.x)
    env.step(dt, u)

env.plot()

agent.plot()

plt.show()

print(env.history[0])

print(env.history[1])
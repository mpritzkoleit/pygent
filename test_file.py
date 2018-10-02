import numpy as np
import matplotlib.pyplot as plt
from PyGent import StateSpaceModel, FeedBack

def ode(t, x, u):
    dx1dt =  x[1]
    dx2dt =  x[2]
    dx3dt =  - 8*(u[0] + u[1])
    return [dx1dt, dx2dt, dx3dt]

def mu(x):
    u1 = x[0]
    u2 = x[1] + x[2]
    return [u1, u2]

t0 = 0
tf = 20
dt = 0.05
tt = np.arange(t0, tf, dt)

x0 = [2, 1, -2.5]
env = StateSpaceModel(ode, x0)
m = 2
agent = FeedBack(mu, m)

for t in tt:
    u = agent.take_action(dt, env.x)
    env.step(dt, u)

env.plot()

agent.plot()

plt.show()
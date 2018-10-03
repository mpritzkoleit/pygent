from Environment import StateSpaceModel
from NFQ import NFQ
import numpy as np
import matplotlib.pyplot as plt


def ode(t, x, u):
    g = 9.81  # gravity
    b = 0.0  # dissipation
    u1 = u[0] # torque
    x1, x2 = x
    dx1dt = x2
    dx2dt = u1 + g * np.sin(x1) - b * x2

    return [dx1dt, dx2dt]

def cost(x, u):
    x1, x2 = x
    # map x1 to [-pi,pi]

    if abs(x2) > 10:
        c = 1
    elif abs(x1)<0.1 and abs(x2)<0.1:
        c = 0
    else:
        c = 0.01
    return c

x0 = [np.pi, 0]

pendulum = StateSpaceModel(ode,cost,x0)
t = 3
dt = 0.05
controls = np.array([-1, 1]).T

learner = NFQ(pendulum, controls, t, dt)

#learner.run_episode()
#learner.plot()
learner.run_learning(50)
import numpy as np
import matplotlib.pyplot as plt
from Environments import OpenAIGym, Pendulum
from DDPG import DDPG
from NFQ import NFQ


id = 'Pendulum-v0'
#id = 'MountainCarContinuous-v0'
pendulum = OpenAIGym(id, render=True)
t = 5
dt = 0.05

xDim = pendulum.n
uDim = 1
uMax = 2

t = 10
dt = 0.05
controls = np.array([-2, 0, 2]).T
xGoal = [0, 0]
gamma = 0.99
eps = 0.1
netStructure = [4, 20, 20, 1]


#algorithm = NFQ(pendulum, controls, xGoal, t, dt, netStructure, eps, gamma)
algorithm = DDPG(pendulum, xDim, uDim, uMax, t, dt)

#algorithm.run_episode()
algorithm.run_learning(10000)
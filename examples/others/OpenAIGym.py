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

xDim = pendulum.xDim
uDim = 1
uMax = 1

t = 10
dt = 0.05
controls = np.array([-2, 0, 2]).T

#algorithm = NFQ(pendulum, controls, xGoal, t, dt, netStructure, eps, gamma)
algorithm = DDPG(pendulum, t, dt, path = '../Controllers/DDPG/Pendulum-Gym/')

#algorithm.run_episode()
algorithm.run_learning(10000)
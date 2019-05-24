from pygent.environments import OpenAIGym
from pygent.algorithms.ddpg import DDPG

#id = 'Pendulum-v0'
id = 'MountainCarContinuous-v0'
pendulum = OpenAIGym(id, render=True)

t = 10
dt = 0.05

path = '../../../results/gym/experiment2/'


algorithm = DDPG(pendulum,t, dt, path=path, warm_up=64)

algorithm.run_learning(10000)
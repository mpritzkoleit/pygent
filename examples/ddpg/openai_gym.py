from pygent.environments import OpenAIGym
from pygent.algorithms.ddpg import DDPG

id = 'Pendulum-v0'
#id = 'MountainCarContinuous-v0'

env = OpenAIGym(id, render=True) #instanciate the gym environment

t = 10 # time of an episode
dt = 0.05 # time step-size

path = '../../../results/gym/' # path, where results are saved

algorithm = DDPG(env ,t , dt, path=path, warm_up=64) # instance of the DDPG algorithm

learning_steps = 100000 # define training duration
algorithm.run_learning(learning_steps) # run reinforcment learning
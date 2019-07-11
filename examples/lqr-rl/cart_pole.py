from pygent.environments import CartPole
from pygent.algorithms.ddpg import DDPG
from pygent.algorithms.ilqr import iLQR
from pygent.data import DataSet
from pygent.helpers import mapAngles
import numpy as np
import torch
import torch.nn as nn
# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = 0.5*x1**2 + x2**2 + 0.02*x3**2 + 0.05*x4**2 + 0.05*u1**2
    return c

def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 100*x2**2 + 10*x3**2 + 10*x4**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.01, 0.01), np.random.uniform(0.99*np.pi, 1.01*np.pi), 0, 0]
    return x0



t = 10 # time of an episode
dt = 0.02 # time step-size

env = CartPole(c_k, p_x0, dt)
env2 = CartPole(c_k, p_x0, dt)

env2.terminal_cost = 200 # define the terminal cost if x(k+1) is a terminal state

path = '../../../results/cart_pole/lqr-rl/'  # path, where results are saved

algorithm = iLQR(env, 6, dt, path=path, fcost=c_N, constrained=True, dataset_size=1e4, maxIters=50) # instance of the iLQR algorithm

rl_algorithm = DDPG(env2, t, dt, path=path, warm_up=0, actor_lr=1e-3, plotInterval=10, costScale=1) # instance of the DDPG algorithm

algorithm.run_optim() # run trajectory optimization


R = DataSet(1e6)
D = DataSet(1e6)
for transition in algorithm.R.data:
    x_ = transition['x_']
    o_ = transition['o_']
    o = transition['o']
    u = transition['u']
    t = env2.terminate(x_)
    c = c_k(mapAngles([0,1,0,0],x_), u)*dt
    trans = ({'x_': o_, 'u': u, 'x': o,
                   'c': [c], 't': [False]})
    R.force_add_sample(trans)
    trans = ({'x': o_, 'u': u})
    D.force_add_sample(trans)
pretrain_actor(rl_algorithm.agent.actor1, D)
rl_algorithm.agent.blend_hard(rl_algorithm.agent.actor1, rl_algorithm.agent.actor2)
rl_algorithm.R = R
#algorithm.load() # can be used to load existing networks and data set
#for _ in range(10000):
#    rl_algorithm.agent.training(rl_algorithm.R)
learning_steps = 5e5 # define training duration
rl_algorithm.run_learning(learning_steps) # run reinforcment learning


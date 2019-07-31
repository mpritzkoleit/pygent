from pygent.environments import Pendulum
from pygent.algorithms.mbrl import MBRL
import numpy as np
import time
# define the incremental cost


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--time_step", type=float, default=0.02)
parser.add_argument("--use_mpc", type=bool, default=False)
parser.add_argument("--warm_up_episodes",type=int,  default=10)
parser.add_argument("--agg", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--weight_decay", type=float, default=1e-3)
args = parser.parse_args()

def c_k(x, u):
    x1, x2 = x
    u1, = u
    c = x1**2 + .1*x2**2 + 0.01*u1**2
    return c

def c_N(x):
    x1, x2 = x
    c = 100*x1**2 + 1*x2**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(0.99*np.pi, 1.01*np.pi), np.random.uniform(-.01, .01)]
    return x0


x0 = [np.pi, 0]
t = 6 # time of an episode
dt = args.time_step # time step-size

env = Pendulum(c_k, x0, dt)

env.uMax = env.uMax

path = '../../../results/mbrl_'+str(int(time.time()))+'/'  # path, where results are saved
rl_algorithm = MBRL(env, t, dt,
                    path=path,
                    horizon=3.,
                    fcost=c_N,
                    warm_up_episodes=args.warm_up_episodes,
                    use_mpc=args.use_mpc,
                    ilqr_print=False,
                    ilqr_save=False,
                    aggregation_interval=args.agg,
                    training_epochs=args.epochs,
                    weight_decay=args.weight_decay)

rl_algorithm.run_learning(50)

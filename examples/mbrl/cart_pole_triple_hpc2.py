from pygent.environments import CartPoleTriple
from pygent.algorithms.mbrl import MBRL

import numpy as np
import matplotlib
matplotlib.use('Agg') # disable interactive display of figures on the HPC-cluster
# define the incremental cost

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_id", type=int, default=0)
parser.add_argument("--time_step", type=float, default=0.002)
parser.add_argument("--use_mpc", type=int, default=0)
parser.add_argument("--warm_up_episodes",type=int,  default=3)
parser.add_argument("--agg", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--data_noise", type=float, default=1e-3)
parser.add_argument("--path", type=str, default=1e-3)
parser.add_argument("--data_set", type=str, default='')
parser.add_argument("--episodes", type=int, default=50)
args = parser.parse_args()

# define the incremental cost
def c_k(x, u, mod):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    u1, = u
    c = 15*x1**2 + 10*(x2-mod.pi)**2 + 10*(x3-mod.pi)**2 + 10*(x4-0*mod.pi)**2 + .1*u1**2
    return c

# define the final cost at step N
def c_N(x, mod):
    x1, x2, x3, x4, x5, x6, x7, x8 = x
    c = 1000*x1**2 + 1000*(x2-mod.pi)**2 + 1000*(x3-mod.pi)**2 + 1000*(x4-0*mod.pi)**2 + 1000*x5**2 + 1000*x6**2 + 1000*x7**2 + 1000*x8**2
    return c

# initial state value
x0 = [0, np.pi, np.pi, np.pi, 0, 0, 0, 0]

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.005, 0.005), np.pi, np.pi, np.pi, 0, 0, 0, 0]
    return x0

t = 3.5 # time of an episode
dt = args.time_step # time step-size

env = CartPoleTriple(c_k, p_x0, dt)

path = args.path + str(args.exp_id)+'/' 

rl_algorithm = MBRL(env, t, dt,
                    path=path,
                    horizon=2.,
                    fcost=c_N,
                    warm_up_episodes=args.warm_up_episodes,
                    use_mpc=args.use_mpc,
                    ilqr_print=False,
                    ilqr_save=False,
                    aggregation_interval=args.agg,
                    training_epochs=args.epochs,
                    weight_decay=args.weight_decay,
                    data_noise=args.data_noise)

if args.data_set != '':
    rl_algorithm.D_rand.load(args.data_set)
rl_algorithm.run_learning(args.episodes)


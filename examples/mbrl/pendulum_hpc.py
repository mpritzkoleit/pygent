from pygent.environments import Pendulum
from pygent.algorithms.mbrl import MBRL
import numpy as np
import time
import matplotlib
matplotlib.use('Agg') # disable interactive display of figures on the HPC-cluster
# define the incremental cost

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--test_t", type=float, default=0.)
parser.add_argument("--t", type=float, default=6.)
parser.add_argument("--exp_id", type=int, default=0)
parser.add_argument("--time_step", type=float, default=0.05)
parser.add_argument("--use_mpc", type=int, default=0)
parser.add_argument("--warm_up_episodes",type=int,  default=1)
parser.add_argument("--agg", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--data_noise", type=float, default=1e-3)
parser.add_argument("--path", type=str, default='./')
parser.add_argument("--episodes", type=int, default=50)
parser.add_argument("--pred_err_bound", type=float, default=0.0)
args = parser.parse_args()

def c_k(x, u, t, mod):
    x1, x2 = x
    u1, = u
    c = x1**2 + .01*x2**2 + 0.5*(1000*mod.exp(-t*15)+ 1)*u1**2
    return c

def c_N(x):
    x1, x2 = x
    c = 100*x1**2 + 10*x2**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(0.99*np.pi, 1.01*np.pi), np.random.uniform(-.01, .01)]
    return x0


x0 = [np.pi, 0]

t = args.t # time of an episode

dt = args.time_step # time step-size

env = Pendulum(c_k, x0, dt)

env.uMax = env.uMax/3.5*5

path = args.path + str(args.exp_id)+'/'

rl_algorithm = MBRL(env, t, dt,
                    path=path,
                    horizon=t,
                    fcost=c_N,
                    test_t=args.test_t,
                    warm_up_episodes=args.warm_up_episodes,
                    use_mpc=args.use_mpc,
                    ilqr_print=True,
                    ilqr_save=True,
                    aggregation_interval=args.agg,
                    training_epochs=args.epochs,
                    weight_decay=args.weight_decay,
                    data_noise=args.data_noise,
                    prediction_error_bound=args.pred_err_bound,
                    maxIters=100)
rl_algorithm.run_learning(args.episodes)




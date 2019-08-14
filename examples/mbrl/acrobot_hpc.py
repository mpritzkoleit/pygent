from pygent.environments import Acrobot
from pygent.algorithms.mbrl import MBRL

import numpy as np
import matplotlib
matplotlib.use('Agg') # disable interactive display of figures on the HPC-cluster
# define the incremental cost

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--time_step", type=float, default=0.03)
parser.add_argument("--use_mpc", type=int, default=0)
parser.add_argument("--warm_up_episodes",type=int,  default=5)
parser.add_argument("--agg", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--data_noise", type=float, default=1e-3)
parser.add_argument("--pred_err_bound", type=float, default=1e-2)
args = parser.parse_args()

# define the incremental cost
# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = 5*x1**2 + x2**2 + 0.01*x3**2 + 0.01*x4**2 + 0.005*u1**2
    return c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 10*x2**2 + 10*x3**2 + 10*x4**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(0.99*np.pi, 1.01*np.pi), np.random.uniform(-0.01, 0.01), 0, 0]
    return x0


t = 6 # time of an episode
dt = args.time_step # time step-size

env = Acrobot(c_k, p_x0, dt, linearized=False)
env.uMax = env.uMax*0.7

path = '/scratch/p_da_reg/results/mbrl/acrobot/'+'mpc='+str(args.use_mpc)+'/'+'weight_decay='+str(args.weight_decay)+'/'

rl_algorithm = MBRL(env, t, dt,
                    path=path,
                    horizon=2.,
                    fcost=c_N,
                    warm_up_episodes=args.warm_up_episodes,
                    use_mpc=args.use_mpc,
                    ilqr_print=True,
                    ilqr_save=False,
                    aggregation_interval=args.agg,
                    training_epochs=args.epochs,
                    weight_decay=args.weight_decay,
                    data_noise=args.data_noise,
                    prediction_error_bound=args.pred_err_bound,
                    maxIters=200)

rl_algorithm.run_learning(500)


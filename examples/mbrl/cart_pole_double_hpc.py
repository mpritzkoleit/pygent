from pygent.environments import CartPoleDoubleSerial
from pygent.algorithms.mbrl import MBRL

import numpy as np
import matplotlib
matplotlib.use('Agg') # disable interactive display of figures on the HPC-cluster
# define the incremental cost

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--time_step", type=float, default=0.01)
parser.add_argument("--use_mpc", type=int, default=0)
parser.add_argument("--warm_up_episodes",type=int,  default=3)
parser.add_argument("--agg", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--data_noise", type=float, default=1e-3)
args = parser.parse_args()

# define the incremental cost
def c_k(x, u):
    x1, x2, x3, x4, x5, x6 = x
    u1, = u
    c = 2*x1**2 + 3*x2**2 + 2*x3**2 + 0.02*x4**2 + 0.03*x5**2 + 0.03*x6**2 + 0.01*u1**2
    return c

# define the final cost at step N
def c_N(x):
    x1, x2, x3, x4, x5, x6 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2 + 10*x4**2 + 10*x5**2 + 10*x6**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.01, 0.01), np.random.uniform(0.99*np.pi, 1.01*np.pi),
        np.random.uniform(0.99*np.pi, 1.01*np.pi), 0, 0, 0]
    return x0


x0 = [0, np.pi, np.pi, 0, 0, 0]
t = 5 # time of an episode
dt = args.time_step # time step-size

env = CartPoleDoubleSerial(c_k, p_x0, dt)

path = '/scratch/p_da_reg/results/mbrl/cart_pole_double/'+'time_step='+str(args.time_step)+'/'+'weight_decay='+str(args.weight_decay)+'/'

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

rl_algorithm.load()
rl_algorithm.run_learning(50)


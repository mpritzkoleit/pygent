import numpy as np
import scipy.optimize as optim
import argparse
import matplotlib
matplotlib.use('Agg') # disable interactive display of figures on the HPC-cluster

# pygent imports
from pygent.environments import CartPole
from pygent.algorithms.mbrl import MBRL

parser = argparse.ArgumentParser()
parser.add_argument("--time_step", type=float, default=0.02)
parser.add_argument("--use_mpc", type=int, default=0)
parser.add_argument("--warm_up_episodes",type=int,  default=3)
parser.add_argument("--agg", type=int, default=1)
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--pred_err_bound", type=float, default=1e-2)
args = parser.parse_args()

# incremental cost
def c_k(x, u):
    x1, x2, x3, x4 = x
    u1, = u
    c = 2*x1**2 + 5*x2**2 + .01*x3**2 + .01*x4**2 + 0.1*u1**2
    return c

# final cost at timestep N
def c_N(x):
    x1, x2, x3, x4 = x
    c = 100*x1**2 + 100*x2**2 + 100*x3**2 + 100*x4**2
    return c

# define the function, that represents the initial value distribution p(x_0)
def p_x0():
    x0 = [np.random.uniform(-0.005, 0.005), np.pi, 0, 0]
    return x0


x0 = [0, np.pi, 0, 0] # initial value
t = 5 # duration of an episode in seconds
dt = args.time_step # time step-size

# initialize the system 
sys = CartPole(c_k, p_x0, dt)

path = '../results/mbrl/'  # path, where results are saved'

# instance of the model-based RL algorithm 
rl_algorithm = MBRL(sys, t, dt,
                    path=path,
                    fcost=c_N,
                    warm_up_episodes=args.warm_up_episodes,
                    use_mpc=args.use_mpc,
                    ilqr_print=True,
                    ilqr_save=False,
                    aggregation_interval=args.agg,
                    training_epochs=args.epochs,
                    weight_decay=args.weight_decay,
                    prediction_error_bound=args.pred_err_bound,
                    test_t=5.0)

rl_algorithm.run_learning(50)


